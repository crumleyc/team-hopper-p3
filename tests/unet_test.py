from src.unet import unet

unet = UNet()

def dice_coef_test():
    #Creating testing arrays
    y_true = np.array([[1,2,3],[1,2,3],[1,2,3]])
    y_pred = np.array([[3,2,1],[3,2,1],[3,2,1]])

    dc = unet.dice_coef(y_true,y_pred)
    #Correct Answer is 2*3/(9+9)=6/18=1/3
    assert dc == (1/3)


def dice_coef_loss_test():
    #Creating testing arrays
    y_true = np.array([[1,2,3],[1,2,3],[1,2,3]])
    y_pred = np.array([[3,2,1],[3,2,1],[3,2,1]])

    dcl = dice_coef_loss(y_true,y_pred)
    #Correct Answer is -(2*3/(9+9))=-(6/18)=-(1/3)
    assert dcl == -(1/3)
