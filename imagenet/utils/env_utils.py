IMAGENET_PATH="./data/imagenet-1k/train"
CELEBA_PATH="./data/celeba_hq_256"
PARTIPROMPOTS_PATH="./data/partiprompts_1000.txt"
WIKIART_PATH='./data/wikiart'
MODEL_PATH='./ckpts'    # Your model path
COND_VALIDITY_PATH_MAPPING = {
    'google/vit-base-patch16-224': "facebook/deit-small-patch16-224",
    "facebook/deit-small-patch16-224": 'google/vit-base-patch16-224',
    'resnet_cifar10.pt': "ahsanjavid/convnext-tiny-finetuned-cifar10",
    'ozzyonfire/bird-species-classifier': "chriamue/bird-species-classifier",
    "timeclassifier_cifar10.pt": "ahsanjavid/convnext-tiny-finetuned-cifar10",
    "timeclassifier_imagenet.pt": "google/vit-base-patch16-224",
    'nateraw/vit-age-classifier': 'ibombonato/swin-age-classifier', # 0 for young, 1 for old
    'rizvandwiki/gender-classification-2': 'rizvandwiki/gender-classification', # 0 for female 1 for male
    'enzostvs/hair-color': 'londe33/hair_v02',   # hair color (3->1 red, 0->2 black, 1->3 blond)
    'chriamue/bird-species-classifier': "dennisjooo/Birds-Classifier-EfficientNetB2",
    'dennisjooo/Birds-Classifier-EfficientNetB2': 'chriamue/bird-species-classifier',
}
