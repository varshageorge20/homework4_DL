import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    ret = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    diff = heatmap - (ret > heatmap).float() * 1e4
    k=(heatmap - (ret > heatmap).float() * 1e4).numel()

    if max_det > k:
        max_det = k
    score, va = torch.topk(diff.view(-1), max_det)
    peak = []
    for p, m in zip(score.cpu(), va.cpu()):
        if p > min_score :
            peak.append((float(p), int(m) % heatmap.size(1), int(m) //heatmap.size(1)))
    return peak
    
class Detector(torch.nn.Module):
    def __init__(self, num_classes=3): #3 output classes 
        """
           Your code here.
           Setup your detection network
        """
        super(Detector, self).__init__()
        
        # Define the UNet architecture
        # You can start with the down-sampling (encoder) part
        # self.encoder = nn.Sequential(
        #     # Block 1
        #     nn.Conv2d(3, 16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # Block 2
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # Add more blocks as needed
        # )
        
        # # Define the up-sampling (decoder) part
        # self.decoder = nn.Sequential(
        #     # Block 1
        #     # nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.ConvTranspose2d(16, num_classes, kernel_size=1),

        #     # Block 1 for class 0
        #     nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Output a single channel heatmap
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(1, 4, kernel_size=1),  # Adjust the number of output channels for your task
        #     nn.ReLU(inplace=True),

        #     # Block 2 for class 1
        #     nn.Conv2d(4, 1, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(1, 4, kernel_size=1),
        #     nn.ReLU(inplace=True),

        #     # # Block 3 for class 2
        #     # nn.Conv2d(16, 1, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(1, 4, kernel_size=1),
        #     # nn.ReLU(inplace=True),
        # )

        # self.size_layer = nn.Sequential(
        #     nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, num_classes, kernel_size=1)  # Adjust output channels for your task
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Continue the encoder part as needed
        )

        # Define the decoder (up-sampling) part of the U-Net
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)  # Adjust output channels for your task
        )

        self.size_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)  # Adjust output channels for your task
        )

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
         # Implement forward pass through the network
        x = self.encoder(x)
        object_detection_heatmap = self.decoder(x)
        size_heatmap = self.size_layer(x)  # Define this layer in your model
        return object_detection_heatmap, size_heatmap

    def detect(self, images):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        #ORIGINAL
        # max_detections_per_class = 30
        # detections = []
        # for class_idx in range(3):  # Assuming 3 classes
        #     detections_class = []
        #     for channel in image:
        #         heatmap = self(channel)
        #         detected_objects = extract_peak(heatmap)
        #         for score, cx, cy in detected_objects:
        #             w, h = 0.0, 0.0  # Replace with actual width and height prediction
        #             detections_class.append((score, cx, cy, w / 2, h / 2))
        #     # Limit the number of detections per class
        #     detections_class = sorted(detections_class, reverse=True)[:max_detections_per_class]
        #     detections.append(detections_class)
        # return detections

        max_detections_per_class = 30
        detections_batch = []

        for image in images:
            if image.shape[1] == 1:  # Check if it's a single channel image
                image = torch.cat((image, image, image), dim=1)  
            detections = []
            for class_idx in range(3):  # Assuming 3 classes
                detections_class = []
                heatmap = self(image[class_idx].unsqueeze(0))  # Pass a single channel (class heatmap)
                detected_objects = extract_peak(heatmap)
                for score, cx, cy in detected_objects:
                    w, h = 0.0, 0.0  # Replace with actual width and height prediction
                    detections_class.append((score, cx, cy, w / 2, h / 2))
                # Limit the number of detections per class
                detections_class = sorted(detections_class, reverse=True)[:max_detections_per_class]
                detections.append(detections_class)

            detections_batch.append(detections)

        return detections_batch


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
