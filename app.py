from fastapi import FastAPI, Request,File, UploadFile, Form
from pydantic import BaseModel
import io
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
import tempfile
import base64, json
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
# Allow CORS from any origin (for testing purposes, you can specify specific origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can limit this to specific domains if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index_new.html", {"request": request})

@app.post("/process_image")
# async def process_image(reference_image: UploadFile = File(...), target_image: UploadFile = File(...), bboxes: str = '', classes: str = ''):
async def process_image(
    reference_image: UploadFile = File(...),  # Image file for reference
    target_image: UploadFile = File(...),     # Image file for target
    bboxes: str = Form(...),           # Bounding boxes in JSON format
    classes: str = Form(...),          # Class names in JSON format
    class_names: str = Form(...)
):
    try:
        reference_img_data = await reference_image.read()
        target_img_data = await target_image.read()
        
        # Open the images using PIL
        ref_image = Image.open(io.BytesIO(reference_img_data))
        target_image = Image.open(io.BytesIO(target_img_data))

        # Convert RGBA → RGB if needed
        if ref_image.mode == "RGBA":
            ref_image = ref_image.convert("RGB")
        if target_image.mode == "RGBA":
            target_image = target_image.convert("RGB")
        # ref_image.show()  # Only show the first 100 characters for readability
        # target_image.show()
        
        # Convert class and bbox data from JSON strings
        # Parse string inputs
        bboxes = np.array(eval(bboxes))
        classes = np.array(eval(classes))
        class_names = eval(class_names)  # or json.loads(class_names) if sent as JSON

        # Convert XYWH to XYXY
        xyxy_boxes = []
        for box in bboxes:
            x, y, w, h = box
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            xyxy_boxes.append([x1, y1, x2, y2])

        bboxes = np.array(xyxy_boxes)

        # print(f"Bboxes: {bboxes}")
        # print(f"Classes: {classes}")
        # Save the images temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as ref_tmp, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tgt_tmp:
            
            ref_image.save(ref_tmp.name)
            target_image.save(tgt_tmp.name)
            
            visual_prompts = {
                "bboxes": bboxes,
                "cls": classes
            }
            print(bboxes)
            print(classes)
            print(visual_prompts)
            print(tgt_tmp.name)
            print(ref_tmp.name)
            # Initialize YOLO model
            model = YOLOE("yoloe-11s-seg.pt")
            print("model loadeddddddddd")
            # Run prediction
            results = model.predict(
                tgt_tmp.name,
                refer_image=ref_tmp.name,
                visual_prompts=visual_prompts,
                predictor=YOLOEVPSegPredictor,  # Include if required
                save=True,
                augment=True
            )
            print("results came",results[0])
            for i, box in enumerate(results[0].boxes.data):
                cls_id = int(box[5])
                results[0].names[cls_id] = class_names[cls_id]
            # results[0].show()
            detected_info = []
            class_counts = {}

            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls_id = box.tolist()
                cls_id = int(cls_id)
                class_name = class_names[cls_id]

                detected_info.append(
                    f"Detected: {class_name} - Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) - Confidence: {conf:.2f}"
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            print("Detected results")
            print(detected_info)
            # return JSONResponse(content=class_counts)

            # result_img = results[0].plot()  # result_img is a NumPy array (HWC, RGB)
            # img_pil = Image.fromarray(result_img)
            # buffered = BytesIO()
            # img_pil.save(buffered, format="PNG")
            # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            result_img = results[0].plot()  # still RGB
            # Swap Red and Blue channels manually (BGR->RGB or vice-versa)
            # Swap channels from RGB to BGR manually
            
            

            # Encode image as PNG using OpenCV (to retain the BGR pixel order)
            _, buffer = cv2.imencode('.png', result_img)  # IMPORTANT: Use OpenCV not PIL here!
            img_str = base64.b64encode(buffer).decode('utf-8')
            # # Convert RGB to BGR
            # result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

            # # Encode as PNG
            # _, buffer = cv2.imencode('.png', result_img_bgr)
            # img_str = base64.b64encode(buffer).decode("utf-8")

            return JSONResponse(content={
                "image_base64": img_str,
                "detection_counts": class_counts,
                "detection_summary": detected_info
            })

    except Exception as e:
        print("errorrrrrrrr",e)
        return JSONResponse(status_code=400, content={"error": str(e)})

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
