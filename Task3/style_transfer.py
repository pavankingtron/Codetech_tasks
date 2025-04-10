
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')

    size = max_size if max(image.size) > max_size else max(image.size)

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Load content and style images
content = load_image("content.jpg").to(device)
style = load_image("style.jpg").to(device)

# Load pretrained VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Content and Style layer names
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Get features
def get_features(image, model, layers):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            name = f"conv{i}_{1}"
            if name in layers:
                features[name] = x
            i += 1
    return features

# Gram Matrix for style representation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Extract features
content_features = get_features(content, vgg, content_layers + style_layers)
style_features = get_features(style, vgg, content_layers + style_layers)

# Style Gram Matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Start with a copy of the content image
target = content.clone().requires_grad_(True).to(device)

# Define optimizer
optimizer = torch.optim.Adam([target], lr=0.003)
style_weight = 1e6
content_weight = 1

# Run optimization
for step in range(1, 201):
    target_features = get_features(target, vgg, content_layers + style_layers)

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.4f}")

# Unnormalize and save output
def save_image(tensor, filename):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)

save_image(target, "output.jpg")
print("Style Transfer Complete! Output saved as 'output.jpg'.")
