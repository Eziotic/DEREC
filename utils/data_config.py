from .util_func import ThinLenCamera
import numpy as np

def get_data_config(args):
    if args.dataset == 'NYUv2':
        if args.normalize_dpt:
            fd = [0.2, 0.3, 0.4, 0.65, 0.95]
        else:
            fd = [1, 1.5, 2.5, 4, 6]

        dataset_config = {
            'root_dir': args.data_path,
            'norm': args.normalize_dpt, 
            'shuffle': args.shuffle,
            'img_num': args.image_num, 
            'visible_img': args.visible_image_num,
            'focus_dist': fd,
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'scale': 1,
            'near': args.camera_near,
            'far': args.camera_far,
        }

    elif args.dataset == 'NYU100':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': args.shuffle,
            'img_num': 100, 
            'visible_img': args.visible_image_num,
            'focus_dist': np.linspace(1, 9, 100),
            'recon_all': False,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'scale': 1,
            'near': args.camera_near,
            'far': args.camera_far,
        }

    elif args.dataset == 'DSLR':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': args.shuffle,
            'img_num':  5, 
            'visible_img': 5,
            'focus_dist': [1, 1.5, 2.5, 4, 6],
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'DPT': args.DPT,
            'AIF': args.AIF,
            'near': args.camera_near,
            'far':args.camera_far,
            'scale':1,
        }

    elif args.dataset == 'mobileDFD':
        dataset_config = {
            'root_dir': args.data_path,
            'visible_img': args.visible_image_num,
            'recon_all': args.recon_all,
            'RGBFD': args.RGBFD,
            'scale': 1,
            'near': args.camera_near,
            'far': args.camera_far,
        }

    elif args.dataset == 'SC':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': False,
            'img_num':  args.image_num, 
            'visible_img': args.visible_image_num,
            'focus_dist': [0.5, 0.6, 0.75, 1.0, 1.6, 3.0],
            'recon_all': args.recon_all,
            'near': args.camera_near,
            'far':args.camera_far,
            'scale':10,
        }

    elif args.dataset == 'DDFF':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': True,
            'img_num':  10,
            'scale' : args.scale,
            'visible_img': 10,
            'recon_all': args.recon_all,
            'RGBFD': True,
            'DPT': True,
            'AIF': True,
        }
    elif args.dataset == 'defocus':
        dataset_config = {
            'root_dir': args.data_path,
            'shuffle': False,
            'img_num':  5, 
            'RGBFD': True,
            'scale' : args.scale,
            'DPT': True,
            'AIF': False,
}
    else:
        exit()
    return dataset_config

def get_camera(args):
    if args.dataset == 'NYUv2':
        if not args.normalize_dpt:
            camera = ThinLenCamera(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
        else:
            camera = ThinLenCamera(fnumber=0.5, focal_length=2.9*1e-3, pixel_size=5.6e-6)
    elif args.dataset == 'NYU100':
        camera = ThinLenCamera(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
        # print('Modified Camera!!!!!')
        # camera = ThinLenCamera(fnumber=8.0, focal_length=50*1e-3, pixel_size=1.2e-5)
    elif args.dataset == 'DSLR':
        camera = ThinLenCamera(fnumber=1.2, focal_length=17*1e-3, pixel_size=1.2e-5)
    elif args.dataset == 'SC':
        camera = ThinLenCamera(fnumber=args.fnumber, focal_length=12*1e-3, pixel_size=6.5e-5)
    elif args.dataset == 'mobileDFD':
        camera = ThinLenCamera(fnumber=24, focal_length=50*1e-3, pixel_size=5.6e-6)
    elif args.dataset == 'DDFF':
        camera = ThinLenCamera(fnumber=2, focal_length=9.5*1e-3, pixel_size=1.2e-5)
    elif args.dataset == 'defocus':
        camera = ThinLenCamera(fnumber=1, focal_length=2.9 * 1e-3, pixel_size=1.2e-5)
    else:
        camera = ThinLenCamera(args.fnumber, args.focal_length, args.sensor_size, args.image_size)
    return camera



