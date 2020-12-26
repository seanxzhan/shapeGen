import glob
import scipy.io as sio
import pywavefront as pf

def check_data():
    """
    Checks if data can be loaded by scipy.io
    """
    all_mat = glob.glob('.\\02691156\\*.mat')
    # model = sio.loadmat('.\\02691156\\1a6ad7a24bb89733f412783097373bdc.mat')
    for f in all_mat:
        try:
            model = sio.loadmat(f)
        except:
            print("err loading: " + f)
    print("------ FINISHED CHECKING ------")

def convert_to_arrays():
    file_dir = '.\\square_rings\\0.obj'
    scene = pf.Wavefront(file_dir)
    print(scene)

if __name__ == '__main__':
    # check_data()
    # convert_to_arrays()
    model = sio.loadmat('.\\02691156\\1a6ad7a24bb89733f412783097373bdc.mat')
    print(model)