#
# You can modify this files
#
import torch.nn.modules.linear
import torchvision
from torchvision import transforms

class HoadonOCR:
    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']

    #hàm tìm tên hoá đơn khi truyền ảnh hoá đơn vào
    def find_label(self, img):
        classes = ['highlands', 'others', 'phuclong', 'starbucks']

        # GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # resnet18
        model = torchvision.models.resnet18(pretrained=True)

        model.fc = torch.nn.modules.linear.Linear(in_features=512, out_features=4, bias=True)

        # connect GPU
        model.to(device)

        # load model
        model.load_state_dict(torch.load('./model.pth', map_location=device))

        # transforms
        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = img_transforms(img) #ep kieu, chuyển dạng ảnh ve PIL.Image.Image image mode=L size=128x128 at 0x7F2829D85A30

        image = image.unsqueeze(0)

        image = image.to(device)

        model.eval()

        nameOfReceipt = ''

        with torch.no_grad():
            output = model(image)  #
            _, predicted = torch.max(output, dim=1)
            nameOfReceipt = classes[int(predicted.cpu().numpy())]
            #print('ten hoa don: ', nameOfReceipt)

        return nameOfReceipt