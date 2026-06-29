"""Spatial navigation tools v3 for ROS MCP.
Q4: NavDistance(Object,u,v,direction) — pixel-select, code auto-computes dist from depth 3x3 patch.
Q6: nav_distance returns target only, navigation executed by navigator.geo_goto.
Q11: lookup_by_position(bearing,depth_m) — find object by position, ~0.3m tolerance.
"""
import json, math, os, time, base64
import numpy as np
from fastmcp import FastMCP
from ros_mcp.utils.websocket import WebSocketManager, parse_input

DEFAULT_NAMESPACE = os.environ.get("ROS_NAMESPACE", "agent0")
DEFAULT_LIN_SPEED = 0.3; DEFAULT_ANG_SPEED = 0.8
DEFAULT_MEMORY_ROOT = os.environ.get("MEMORY_ROOT", "/home/kuko/Kuko1414/memory_navi/memory")
DEFAULT_ENV_NAME = os.environ.get("ENV_NAME", "sim")
DIR_OFFSET = {"front": 0.0, "left": -math.pi/2, "right": math.pi/2}

def _load_mem(area):
    p=os.path.join(DEFAULT_MEMORY_ROOT,DEFAULT_ENV_NAME,area,"area.json")
    return json.load(open(p)) if os.path.exists(p) else None
def _save_mem(area,rec):
    p=os.path.join(DEFAULT_MEMORY_ROOT,DEFAULT_ENV_NAME,area,"area.json")
    os.makedirs(os.path.dirname(p),exist_ok=True)
    t=p+".tmp";json.dump(rec,open(t,"w"),ensure_ascii=False,indent=2);os.replace(t,p)
def _find_obj(area,hint):
    rec=_load_mem(area)
    if not rec: return None
    for o in rec.get("objects",[]) or []:
        if (hint or "").lower() in (o.get("name") or "").lower():
            ap=o.get("abs_pose")
            if ap and ap.get("x") is not None: return {"name":o["name"],"abs_pose":{"x":ap["x"],"y":ap["y"]}}
    return None
def _qyaw(x,y,z,w): return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
def _sdiff(a,b): return ((a-b+180)%360)-180
def _twist(vx=0,wz=0): return {"linear":{"x":vx,"y":0,"z":0},"angular":{"x":0,"y":0,"z":wz}}

def register_spatial_nav_tools(mcp:FastMCP, ws_manager:WebSocketManager):
    ns=DEFAULT_NAMESPACE; gps_t=f"/{ns}/gps"; imu_t=f"/{ns}/imu"; shm=f"/dev/shm/agent_safety_{ns}.json"
    def _safe():
        try: return bool(json.load(open(shm)).get("tripped"))
        except: return False
    def _pose():
        def g(t,m):
            with ws_manager:
                if ws_manager.send({"op":"subscribe","topic":t,"type":m,"queue_length":1,"throttle_rate":0}): return None
                end=time.time()+2
                try:
                    while time.time()<end:
                        r=ws_manager.receive(0.5)
                        if not r: continue
                        md,_=parse_input(r,False)
                        if md and md.get("op")=="publish" and md.get("topic")==t: return md.get("msg",{})
                    return None
                finally: ws_manager.send({"op":"unsubscribe","topic":t})
        gps=g(gps_t,"geometry_msgs/msg/PointStamped"); imu=g(imu_t,"sensor_msgs/msg/Imu")
        if not gps or not imu: return None
        p=gps.get("point",gps); q=imu.get("orientation",{})
        return {"x":float(p.get("x",0)),"y":float(p.get("y",0)),
                "yaw_deg":_qyaw(float(q.get("x",0)),float(q.get("y",0)),float(q.get("z",0)),float(q.get("w",1)))}

    def _depth_patch(u,v):
        """读取深度图像素(u,v)的3x3邻域中位距离。"""
        t=f"/{ns}/camera/depth/image"
        with ws_manager:
            if ws_manager.send({"op":"subscribe","topic":t,"type":"sensor_msgs/msg/Image","queue_length":1,"throttle_rate":0}): return None
            end=time.time()+2.0
            try:
                while time.time()<end:
                    r=ws_manager.receive(0.5)
                    if not r: continue
                    md,_=parse_input(r,False)
                    if md and md.get("op")=="publish" and md.get("topic")==t:
                        d=md.get("msg",{}).get("data","")
                        if d:
                            raw=np.frombuffer(base64.b64decode(d),dtype=np.float32)
                            h,w=md["msg"]["height"],md["msg"]["width"]
                            depth=raw.reshape(h,w)
                            patch=[depth[y][x] for x in range(max(0,u-1),min(w,u+2))
                                   for y in range(max(0,v-1),min(h,v+2))
                                   if 0.05<depth[y][x]<9.5]
                            return float(np.median(patch)) if patch else None
                return None
            finally: ws_manager.send({"op":"unsubscribe","topic":t})

    # ===== if_in_memory =====
    @mcp.tool(description="Check if object has coordinates in semantic memory.")
    def if_in_memory(object_name:str, area:str="break_room")->dict:
        obj=_find_obj(area,object_name)
        if obj: return {"in_memory":True,"name":obj["name"],"abs_pose":obj["abs_pose"]}
        return {"in_memory":False,"name":object_name,"suggestion":"请调 RecordArea 登记此物体"}

    # ===== lookup_by_position =====
    @mcp.tool(description="Find object by bearing+depth. Returns {id,name} or {is_new:true}. Tolerance ~0.3m.")
    def lookup_by_position(bearing:str, depth_m:float)->dict:
        try: depth_m=float(depth_m); bearing=(bearing or "center").lower()
        except: return {"error":"bearing string, depth_m number"}
        pose=_pose()
        if not pose: return {"error":"no pose"}
        offset={"left":25,"center":0,"right":-25}.get(bearing,0)
        ang=math.radians(pose["yaw_deg"]+offset)
        wx=pose["x"]+depth_m*math.cos(ang); wy=pose["y"]+depth_m*math.sin(ang)
        rec=_load_mem("break_room")
        best=None; best_d=0.5
        for o in (rec.get("objects",[]) if rec else []):
            ap=o.get("abs_pose")
            if not ap or ap.get("x") is None: continue
            d=math.hypot(ap["x"]-wx,ap["y"]-wy)
            if d<best_d: best={"id":o.get("id",""),"name":o["name"],"dist_m":round(d,2)}; best_d=d
        return best if best else {"is_new":True,"approx_pos":{"x":round(wx,3),"y":round(wy,3)}}

    # ===== NavDistance (Q4+Q6) =====
    @mcp.tool(description=(
        "Compute navigation target relative to object. u,v=pixel coords on the object. "
        "direction=left|right|front. Distance auto-computed from depth 3x3 patch. "
        "Returns target coords — caller should use geo_goto to execute navigation."
    ))
    def nav_distance(object_name:str, u:int, v:int, direction:str)->dict:
        try: u=int(u); v=int(v); direction=(direction or "front").lower()
        except: return {"error":"u,v int, direction string"}
        if direction not in DIR_OFFSET: return {"error":f"direction must be {list(DIR_OFFSET.keys())}"}
        obj=_find_obj("break_room",object_name)
        if not obj: return {"ok":False,"reason":"not_in_memory","object_name":object_name,
                             "suggestion":"请先调 RecordArea 登记此物体"}
        dist_m=_depth_patch(u,v)
        if not dist_m: return {"ok":False,"reason":"depth_read_failed","pixel":[u,v],
                                "suggestion":"该像素无有效深度，换物体上另一个像素"}
        heading=math.radians(180)+DIR_OFFSET[direction]
        tx=obj["abs_pose"]["x"]+dist_m*math.cos(heading)
        ty=obj["abs_pose"]["y"]+dist_m*math.sin(heading)
        return {"ok":True,"status":"target_computed","target":{"x":round(tx,2),"y":round(ty,2)},
                "dist_used_m":round(dist_m,2),"pixel":[u,v]}

    # ===== NavObject =====
    @mcp.tool(description="Navigate to open space between two known objects.")
    def nav_object(obj_a:str, obj_b:str, standoff_m:float=2.0)->dict:
        a=_find_obj("break_room",obj_a); b=_find_obj("break_room",obj_b)
        if not a: return {"ok":False,"reason":"not_in_memory","object_name":obj_a}
        if not b: return {"ok":False,"reason":"not_in_memory","object_name":obj_b}
        mx,my=(a["abs_pose"]["x"]+b["abs_pose"]["x"])/2,(a["abs_pose"]["y"]+b["abs_pose"]["y"])/2
        dx,dy=b["abs_pose"]["x"]-a["abs_pose"]["x"],b["abs_pose"]["y"]-a["abs_pose"]["y"]
        length=math.hypot(dx,dy) or 1
        tx,ty=mx+standoff_m*(-dy/length),my+standoff_m*(dx/length)
        return {"ok":True,"status":"target_computed","target":{"x":round(tx,2),"y":round(ty,2)}}

    # ===== RecordArea =====
    @mcp.tool(description="Register new objects. Qwen fills id/name/confidence. Code fills coords later.")
    def record_area(objects_spec:str)->dict:
        import datetime
        try: objs=json.loads(objects_spec) if isinstance(objects_spec,str) else objects_spec
        except: return {"error":"objects_spec must be valid JSON array"}
        area="break_room"
        rec=_load_mem(area) or {"area":area,"type":"","summary":"","objects":[]}
        for o in (objs or []):
            if not isinstance(o,dict) or not o.get("name"): continue
            rec["objects"].append({"id":o.get("id",""),"name":o["name"],
                "confidence":float(o.get("confidence",0.6)),"abs_pose":None})
        rec["observed_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        _save_mem(area,rec)
        return {"ok":True,"registered":len(objs or []),
                "note":"abs_pose 将在下轮观测时由系统自动补全"}
