import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import torch
from preprocess import visualize_scatterplot, visualize_3d_arr
from sklearn.manifold import TSNE
from chamferdist import ChamferDistance
import mcubes
from sklearn.metrics import mean_squared_error

def show_gan_loss():    
    """Return a graph of GAN training loss
    """
    gen_L = np.load('./results_latent_32/gen_L.npy')
    disc_L = np.load('./results_latent_32/disc_L.npy')

    plt.plot(gen_L, label="G Loss")
    plt.plot(disc_L, label="D Loss")
    plt.xlabel('Epoch')
    plt.ylim([-2, 5])
    plt.legend()
    plt.show()

def normalize_coors_and_sample(data_path, dataset_size, sample_size, truth, starting_num=0):
    """Normalize coordinates and sample

    Args:
        data_path (String): Directory that stores the coordinates
        dataset_size (Int): Number of objects
        sample_size (Int): Number of objects to sample
        truth (Bool): Whether the input data is ground truth

    Returns:
        List: list containing numpy array of coordinates, each element has
              dimension (num_of_coors, 3)
    """
    if os.path.exists(data_path):
        if truth:
            sampled_idx = np.arange(starting_num, starting_num + sample_size)
        else:
            sampled_idx = np.random.choice(dataset_size, sample_size)
        sampled_coors = []
        counter = 0
        for idx in sampled_idx:
            print(idx)
            counter += 1
            # load .npy files
            if truth:
                arr = np.load(data_path + '/' + str(idx) + '.npy')
                vertices, _ = mcubes.marching_cubes(arr, 0.5)
                coors = vertices # use the vertices' coordinates
            else:
                # these are also coordinates of vertices
                coors = np.load(data_path + '/' + 'out' + str(idx) + '.npy')

            # normalize the coordinates
            norm_min = 0
            norm_max = 1
            scaled_unit = (norm_max - norm_min) / (np.max(coors) - np.min(coors))
            coors = coors*scaled_unit - np.min(coors)*scaled_unit + norm_min
            # print("Normalized coordinates, min:{}, max:{}, shape:{}".format(np.min(coors), np.max(coors), coors.shape))
            print("processed truth:{}, sample #{}".format(truth, counter))
            # coors_v = np.transpose(coors)
            # visualize_scatterplot(coors_v)

            sampled_coors.append(coors)
        return sampled_coors
    else:
        print("{} not found".format(data_path))
        exit(0)

def plot_tsne(ae, gan):
    """Graphs tSNE to compare autoencoder latent code space vs implicit GAN latent code space

    Args:
        ae (np.array): autoencoder latent code
        lgan (np.array): implicit GAN latent code
    """
    tsne = TSNE(n_components=3)
    trans_ae = tsne.fit_transform(ae)
    trans_gan = tsne.fit_transform(gan)
    print(trans_ae.shape)
    print(trans_gan.shape)
    plt.scatter(trans_ae[:, 0], trans_ae[:, 1], c='tab:blue', label='AE Latent Code')
    plt.scatter(trans_gan[:, 0], trans_gan[:, 1], c='tab:orange', label='IMGAN Latent Code')
    plt.xlabel('t-SNE')
    plt.legend()
    plt.show()

def calc_chamfer(truth, generated):
    """Calculates the chamfer distance between truth and generated

    Args:
        truth (List): List containing numpy arrays representing coordinates, 
                      each element corresponds to a ground truth object
        generated (List): List containing numpy arrays representing coordinates, 
                          each element corresponds to a generated object

    Returns:
        Int: Average chamfer distance and average root mean square deviation
    """
    truth_len = len(truth)
    gen_len = len(generated)
    assert truth_len == gen_len

    total_cham = 0

    total_rmse = 0

    for i in range(truth_len):
        print(i)
        t = torch.tensor(truth[i].astype('float32'))
        g = torch.tensor(generated[i].astype('float32'))
        t = torch.reshape(t, (1, t.size()[0], t.size()[1]))
        g = torch.reshape(g, (1, g.size()[0], g.size()[1]))
        chamferdist = ChamferDistance()
        c_dist = 0.5 * chamferdist(g, t, bidirectional=True)
        c_dist = 0.5 * chamferdist(t, g, bidirectional=True)
        cham = c_dist.detach().cpu().item()
        total_cham += cham
        t_verts = truth[i]
        g_verts = generated[i]
        sample_size = 64**3 # sampling resolution
        t_sampled_idx = np.random.choice(t_verts.shape[0], sample_size)
        g_sampled_idx = np.random.choice(g_verts.shape[0], sample_size)
        t_verts = t_verts[t_sampled_idx]
        g_verts = g_verts[g_sampled_idx]
        mse = mean_squared_error(t_verts, g_verts)
        total_rmse += mse
        print("chemfer: {}, mse: {}".format(cham, mse))

    return total_cham / truth_len, total_rmse / truth_len

if __name__ == '__main__':
    # show_gan_loss()

    truth_data_path = './square_rings_vox_64'
    gen_data_path = './gen_square_rings_vox_64'
    truth_set_size = 1000
    test_split = 0.8
    starting_num = int(truth_set_size * test_split)
    # sample_size is used for sampling generated results
    # here we are testing on the test split
    sample_size = truth_set_size - starting_num
    # this (below) should be the same number as ./IMGAN/main.py->number of z vectors generated w/ IMGAN
    gen_set_size = 500
    ae_feature_path = './results_latent_32/data/square_rings_vox64_z.hdf5'
    imgan_feature_path = './results_latent_32/data/IMGAN_z.hdf5'
    calc = True

    if not calc:
        truth_sampled_coors = normalize_coors_and_sample(truth_data_path, truth_set_size, sample_size, truth=True, starting_num=starting_num)
        np.save('./truth_sampled_coors.npy', truth_sampled_coors, allow_pickle=True)
        print("saved truth")
        generated_sampled_coors = normalize_coors_and_sample(gen_data_path, gen_set_size, sample_size, truth=False)
        np.save('./generated_sampled_coors.npy', generated_sampled_coors, allow_pickle=True)
        print("saved generated")
    else:
        truth = np.load('./truth_sampled_coors.npy', allow_pickle=True)
        generated = np.load('./generated_sampled_coors.npy', allow_pickle=True)

        # Compare AE latent code vs. WGAN latent code to see if WGAN learns more general features
        # if points perfectly overlap, there's no reason to use WGAN for such a simple shape
        # tsne: noise -> l-gan (got feature) vs encoder features
        imgan_feature = h5py.File(imgan_feature_path, 'r')
        imgan_z = imgan_feature["gan_z"][:]  # (500,32)
        ae_feature = h5py.File(ae_feature_path, 'r')
        ae_z = ae_feature["zs"][:] # (800,32)
        sampled_idx = np.random.choice(ae_z.shape[0], imgan_z.shape[0])
        ae_z = ae_z[sampled_idx]
        assert ae_z.shape == imgan_z.shape
        plot_tsne(ae_z, imgan_z)

        # t = truth[0]
        # print(t.shape)
        # visualize_scatterplot(np.transpose(t))
        # g = generated[0]
        # print(g.shape)
        # visualize_scatterplot(np.transpose(g))

        # cham: noise -> l-gan -> decoder vs. ground truth
        avg_cham, avg_rmse = calc_chamfer(truth, generated)
        print("average chamfer distance: {}, average rmse: {}".format(avg_cham, avg_rmse))

