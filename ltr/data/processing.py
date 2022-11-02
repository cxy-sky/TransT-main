import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils

import matplotlib.pyplot as plt
import numpy as np


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            template_transform  - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'template': transform if template_transform is None else template_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class TransTProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor,
                 scale_jitter_factor,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

        self.grouth_jittered_size_rand = None
        self.grouth_jittered_center_rand = None

    def _get_jittered_box(self, box, mode, is_grouth=False):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size_rand = torch.randn(2)
        jittered_center_rand = torch.rand(2)

        if self.grouth_jittered_size_rand is None:
            self.grouth_jittered_size_rand = jittered_size_rand
            self.grouth_jittered_center_rand = jittered_center_rand
        if is_grouth:
            jittered_size_rand = self.grouth_jittered_size_rand
            jittered_center_rand = self.grouth_jittered_center_rand
            self.grouth_jittered_size_rand = None
            self.grouth_jittered_center_rand = None
        # h w偏移
        jittered_size = box[2:4] * torch.exp(jittered_size_rand * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        # 中心点偏移
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (jittered_center_rand - 0.5)
        # 再转变为左上角坐标x，y并且拼接h，w
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            if 'search_images_T' in data:
                grouth_images, data['grouth_anno'] = self.transform['joint'](image=data['search_images'],
                                                                             bbox=data['grouth_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 bbox=data['search_anno'])
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'],
                                                                                     bbox=data['template_anno'],
                                                                                     new_roll=False)
            if 'search_images_T' in data:
                data['search_images_T'], data['search_anno_T'] = self.transform['joint'](image=data['search_images_T'],
                                                                                         bbox=data['search_anno_T'])
                data['template_images_T'], data['template_anno_T'] = self.transform['joint'](
                    image=data['template_images_T'],
                    bbox=data['template_anno_T'],
                    new_roll=False)

        # plt.imshow(data['search_images'][0])
        # plt.show()
        # plt.imshow(data['template_images'][0])
        # plt.show()

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 裁剪以jitted_anno框为中心的图像区域 Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        if 'search_images_T' in data:
            for s in ['search', 'template']:
                assert self.mode == 'sequence' or len(data[s + '_images_T']) == 1, \
                    "In pair mode, num search/template frames must be 1"

                # Add a uniform noise to the center pos
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno_T']]

                # 裁剪以jitted_anno框为中心的图像区域 Crop image region centered at jittered_anno box
                if s == 'search':
                    crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images_T'], jittered_anno,
                                                                   data[s + '_anno_T'],
                                                                   self.search_area_factor, self.search_sz)
                elif s == 'template':
                    crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images_T'], jittered_anno,
                                                                   data[s + '_anno_T'],
                                                                   self.template_area_factor, self.temp_sz)
                else:
                    raise NotImplementedError

                # Apply transforms
                data[s + '_images_T'], data[s + '_anno_T'] = self.transform[s](image=crops, bbox=boxes, joint=False)

            jittered_anno = [self._get_jittered_box(a, 'search', is_grouth=True) for a in data['grouth_anno']]
            crops, boxes, _ = prutils.jittered_center_crop(grouth_images, jittered_anno,
                                                           data['grouth_anno'],
                                                           self.search_area_factor, self.search_sz)
            grouth_images, data['grouth_anno'] = self.transform['search'](image=crops, bbox=boxes, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()

        if 'search_images_T' in data:
            data['template_images_T'] = data['template_images_T'].squeeze()
            data['search_images_T'] = data['search_images_T'].squeeze()
            data['template_anno_T'] = data['template_anno_T'].squeeze()
            data['search_anno_T'] = data['search_anno_T'].squeeze()
            data['grouth_anno'] = data['grouth_anno'].squeeze()
            grouth_images = grouth_images[0]

        # show_img(data['search_images'], data['search_anno'], name='search')
        # show_img(data['template_images'], data['template_anno'], name='template')
        #
        # if 'search_images_T' in data:
        #     show_img(grouth_images, data['grouth_anno'], name='grouth')
        #     show_img(data['search_images_T'], data['search_anno_T'], name='search_T')
        #     show_img(data['template_images_T'], data['template_anno_T'], name='template_T')

        return data


def show_img(img, anno, name):
    img = img.numpy().transpose(1, 2, 0) * np.array(
        (0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    plt.imshow(img)
    ax = plt.gca()
    k = anno.numpy()
    ax.add_patch(plt.Rectangle((k[0], k[1]), k[2], k[3], color='blue', fill=False, linewidth=1))
    ax.text(k[0], k[1], name, bbox={'facecolor': 'blue', 'alpha': 0.5})
    plt.show()
