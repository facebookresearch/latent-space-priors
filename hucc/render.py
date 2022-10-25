# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import atexit
import io
import logging
import subprocess
import tempfile
from base64 import b64encode
from typing import List, Optional

import imageio
import numpy as np
import torch as th
import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont, features
from visdom import Visdom

log = logging.getLogger(__name__)

_g_font_path = None


class RenderQueue:
    '''
    An asynchronous queue for plotting videos to visdom.
    '''

    def __init__(self, viz: Optional[Visdom] = None):
        self.viz = viz
        self.queue = mp.Queue()
        self.p = mp.Process(target=self.run, args=(self.queue, viz))
        self.p.start()
        self._call_close = lambda: self.close()
        atexit.register(self._call_close)

    def close(self):
        self.queue.put({'msg': 'quit'})
        self.p.join()
        atexit.unregister(self._call_close)

    def push(
        self,
        img: th.Tensor,
        s_left: List[str] = None,
        s_right: List[str] = None,
    ) -> None:
        self.queue.put(
            {'msg': 'push', 'img': img, 's_left': s_left, 's_right': s_right}
        )

    def plot(self) -> None:
        if self.viz is None:
            raise RuntimeError('No visom instance configured')
        self.queue.put({'msg': 'plot'})

    def save(self, path: str) -> None:
        self.queue.put({'msg': 'save', 'path': path})

    @staticmethod
    def run(queue: mp.Queue, viz: Optional[Visdom] = None):
        imgs = []
        log.debug('Render queue running')
        while True:
            item = queue.get()
            msg = item['msg']
            if msg == 'quit':
                break
            elif msg == 'push':
                imgs.append(item['img'])
                if item['s_left'] or item['s_right']:
                    draw_text(
                        imgs[-1], s_left=item['s_left'], s_right=item['s_right']
                    )
            elif msg == 'plot' and viz and len(imgs) > 0:
                log.debug(f'Plotting video with {len(imgs)} frames to visdom')
                try:
                    plot_visdom_video(viz, imgs)
                except:
                    log.exception('Error plotting video')
                imgs.clear()
            elif msg == 'save':
                log.debug(
                    f'Saving video with {len(imgs)} frames as {item["path"]}'
                )
                try:
                    video_data = video_encode(imgs)
                    with open(item['path'], 'wb') as f:
                        f.write(video_data)
                except:
                    log.exception('Error saving video')
                imgs.clear()


def video_encode(imgs: List[th.Tensor], fps: int = 24):
    '''
    Encode a list of RGB images (HxWx3 tensors) to H264 video, return as a
    binary string.
    '''
    # TODO Can I write directly to a bytesIO object?
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        w = imageio.get_writer(
            tmp.name, format='FFMPEG', mode='I', fps=fps, codec='h264'
        )
        for img in imgs:
            w.append_data(img.numpy())
        w.close()

        data = open(tmp.name, 'rb').read()
    return data


def draw_text(
    img: th.Tensor,
    s_left: List[str] = None,
    s_right: List[str] = None,
    text_color='lime',
):
    '''
    Render text on top of an image (using PIL). Modifies the image
    in-place.
    img: The RGB image (HxWxC)
    s_left: Lines of text, left-aligned, starting from top
    s_right: Lines of text, right-aligned, starting from top
    '''
    global _g_font_path
    if _g_font_path is None:
        _g_font_path = (
            subprocess.check_output(['fc-match', '-f' '%{file}', 'Mono'])
            .decode('utf-8')
            .strip()
        )
    font = _g_font_path
    pimg = Image.fromarray(img.numpy())
    draw = ImageDraw.Draw(pimg)
    fac = 22 if pimg.width >= 400 else 20
    layout_engine = (
        ImageFont.Layout.RAQM
        if features.check('raqm')
        else ImageFont.Layout.BASIC
    )
    fnt = ImageFont.truetype(
        font, pimg.width // fac, layout_engine=layout_engine
    )
    if s_left is not None:
        draw.text(
            (2, 2),
            '\n'.join(s_left),
            fill=text_color,
            stroke_fill='black',
            stroke_width=1,
            font=fnt,
        )
    if s_right is not None:
        _, _, twidth, _ = draw.multiline_textbbox(
            xy=(0, 0), text='\n'.join(s_right), font=fnt, stroke_width=1
        )
        draw.text(
            (pimg.width - twidth - 2, 2),
            '\n'.join(s_right),
            fill=text_color,
            stroke_fill='black',
            stroke_width=1,
            align='right',
            font=fnt,
        )
    a = np.asarray(pimg)
    img.copy_(th.from_numpy(a.copy()))


def plot_visdom_video(
    viz: Visdom, images: List[th.Tensor], show_progress=False, **kwargs
):
    '''
    Plot array of RGB images as a video in Visdom.
    '''
    video_data = video_encode(images)
    encoded = b64encode(video_data).decode('utf-8')
    html = f'<video controls><source type="video/mp4" src="data:video/mp4;base64,{encoded}">Your browser does not support the video tag.</video>'
    viz.text(text=html)
