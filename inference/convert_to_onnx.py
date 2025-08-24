import torch
from model.cnn_model import AnnotatedVocalSetCNN

LABEL_MAPPING = {
        'belt': 0, 'breathy': 1, 'fast_forte': 2, 'fast_piano': 3,
        'forte': 4, 'inhaled': 5, 'lip_trill': 6, 'messa': 7,
        'pp': 8, 'slow_forte': 9, 'slow_piano': 10, 'spoken': 11,
        'straight': 12, 'trill': 13, 'vibrato': 14, 'vocal_fry': 15
    }

model = AnnotatedVocalSetCNN(num_classes=len({v:k for k,v in LABEL_MAPPING.items()}))
model.load_state_dict(torch.load('weights/2025-06-03_00-03/best_model.pth', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 1, 128, 258)
torch.onnx.export(model, dummy_input, "weights/model.onnx")
