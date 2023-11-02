import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from PIL import Image, ImageFilter
import time
import numpy as np
import gradio as gr


def test_model(img):
    weights = torch.load('fine_tuned_weights/resnet50_128_08_100.pt', map_location=torch.device('cpu'))
    model = models.resnet50()
    # Our model outputs the score of DR for classification. See https://arxiv.org/pdf/2110.14160.pdf for more details.
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(weights, strict=True)

    model.eval()

    # Load and preprocess the image
    img = convert(img, 'processed', crop_size=512)

    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(img)

    return output


def ct_model(diseases, img):
    return [{disease: 0.1 for disease in diseases}]


def convert(fname, tgt_path, crop_size=512):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    cropped = cropped.resize([crop_size, crop_size], Image.Resampling.LANCZOS)
    # save(cropped, tgt_path)
    return cropped


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


with gr.Blocks() as demo:
    gr.Markdown(
        """
# Detect Disease From Scan
With this model you can lorem ipsum
- ipsum 1
- ipsum 2
"""
    )
    gr.DuplicateButton()
    disease = gr.CheckboxGroup(
        info="Select the diseases you want to scan for.",
        choices=["Diabetic Retinopathy"], label="Disease to Scan For"
    )
    slider = gr.Slider(0, 100)

    with gr.Tab("X-ray") as x_tab:
        with gr.Row():
            xray_scan = gr.Image()
            xray_results = gr.JSON()
        xray_run = gr.Button("Run")
        xray_run.click(
            test_model,
            inputs=[xray_scan],
            outputs=xray_results,
            api_name="test_model"
        )

    with gr.Tab("CT Scan"):
        with gr.Row():
            ct_scan = gr.Image()
            ct_results = gr.JSON()
        ct_run = gr.Button("Run")
        ct_run.click(
            ct_model,
            inputs=[disease, ct_scan],
            outputs=ct_results,
            api_name="ct_model"
        )

    upload_btn = gr.Button("Upload Results", variant="primary")
    upload_btn.click(
        lambda ct, xr: None,
        inputs=[ct_results, xray_results],
        outputs=[],
    )

if __name__ == "__main__":
    demo.launch()
