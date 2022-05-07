import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import neuralnet_pytorch.gin_nnt as gin
from networks import *
from data_loader import ShapeNet, collate


parser = argparse.ArgumentParser('GraphX-convolution')
parser.add_argument('config_file', type=str, help='config file to dictate training/testing')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
config_file = args.config_file
bs = 150

gin.external_configurable(CNN18Encoder, 'cnn18_enc')
gin.external_configurable(PointCloudEncoder, 'pc_enc')
gin.external_configurable(PointCloudDecoder, 'pc_dec')
gin.external_configurable(PointCloudResDecoder, 'pc_resdec')
gin.external_configurable(PointCloudResGraphXUpDecoder, 'pc_upresgraphxdec')
gin.external_configurable(PointCloudResLowRankGraphXUpDecoder, 'pc_upreslowrankgraphxdec')

def visualize_pc(pc, save_path, i, ground=False):
    fig = plt.figure()
    if isinstance(pc, (list, tuple)) or not len(pc.shape) == 3:
        for ii in range(len(pc)):
            print(ii)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*[pc[ii][:, i] for i in range(pc[ii].shape[-1])], color='blue', alpha=0.6, s=4)
            plt.savefig(os.path.join(save_path, str(i), "ground", str(ii)) + ".jpg")
            plt.clf()
    else:
        for ii in range(pc.shape[0]):
            print(ii)
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = pc[ii, :, 0], pc[ii, :, 1], pc[ii, :, 2]
            ax.scatter(x, y, z, color='red', alpha=0.5, s=5)
            plt.savefig(os.path.join(save_path, str(i), "pred", str(ii)) + ".jpg")
            plt.clf()
    plt.close('all')


@gin.configurable('GraphX')
def test_each_category(data_root, checkpoint_folder, img_enc, pc_enc, pc_dec, color_img=False, n_points=250, **kwargs):
    mon.print_freq, mon.use_tensorboard, mon.use_visdom, mon.current_folder, states = 1, True, True, checkpoint_folder, mon.load('training-91332.pt', method='torch')
    net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 224, 224), (bs, n_points, 3), img_enc, pc_enc, pc_dec)
    print(net)
    net.load_state_dict(states['model_state_dict'])
    net.eval()

    for file_cat in os.listdir(data_root):

        test_data = ShapeNet(path=data_root, grayscale=not color_img, type='test', n_points=n_points)
        test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=10, collate_fn=collate)

        mon.iter = 0
        mon.clear_num_stats(file_cat + '/test chamfer')
        print('Testing...')
        with T.set_grad_enabled(False):
            for itt, batch in mon.iter_batch(enumerate(test_loader)):
                init_pc, image, gt_pc = batch
                if nnt.cuda_available:
                    init_pc = init_pc.cuda()
                    image = image.cuda()
                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc, (list, tuple)) else gt_pc.cuda()

                pred_pc = net(image, init_pc)
                if not os.path.exists("/scratch/ishaanshah/Aman/graphx-conv/src/results/my-model/"):
                    os.makedirs("/scratch/ishaanshah/Aman/graphx-conv/src/results/my-model/")
                np.save(f"/scratch/ishaanshah/Aman/graphx-conv/src/results/my-model/batch_{itt}.npy", pred_pc.cpu().detach().numpy())
                print("Saved batch as numpy array")
                visualize_pc(gt_pc, "/scratch/ishaanshah/Aman/graphx-conv/src/results/my-model/run1/plots/04256520/", str(itt), True)
                visualize_pc(pred_pc, "/scratch/ishaanshah/Aman/graphx-conv/src/results/my-model/run1/plots/04256520/", str(itt), False)
                print("Batch Visualization done")
                loss = sum([normalized_chamfer_loss(pred[None], gt[None]) for pred, gt in zip(pred_pc, gt_pc)]) / (
                            3. * len(gt_pc))
                loss = nnt.utils.to_numpy(loss)
    print('Testing finished!')


if __name__ == '__main__':
    gin.parse_config_file(config_file)
    test_each_category()
