# VGG16 fine-tuning hyperparameters
VGG16_PARAMS = {
    "num_classes": 7,
    "epochs": 5,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "step_size": 3,       # LR scheduler: decay every N epochs
    "gamma": 0.1,         # LR scheduler: multiply LR by this factor
    "freeze_backbone": True,  # Freeze VGG16 feature layers, train only classifier
}
