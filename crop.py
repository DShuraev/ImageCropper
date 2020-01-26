# region [rgba(200,200,200, 0.25)] Imports
import cv2
import numpy

import sys
import pathlib

from PIL import Image
from PIL import ImageCms
from PIL import ImageFile

import argparse
import re
import os
import os.path as path
import fnmatch
# endregion

# buildinfo
build_major = 0
build_minor = 2


# region [rgba(100,50,100, 0.45)] Command line parser
parser = argparse.ArgumentParser()

parser.add_argument('-nlti', help='Don\'t load truncated images', action='store_true', default=False)
parser.add_argument('-m', '--mask', help='File mask', nargs='?', default=argparse.SUPPRESS)
parser.add_argument('-r', '--regex', help='File regex', nargs='?', default=argparse.SUPPRESS)
parser.add_argument('-a', '--all', help='Crop all images using specified scan settings', action='store_true', default=False)
parser.add_argument('-p', '--prefix', help='Output files prefix', nargs='?', default='')
parser.add_argument('-s', '--suffix', help='Output files postfix. Default: _cropped', nargs='?', default='_cropped')
parser.add_argument('-e', '--ext', help='Output extension', nargs='?', default=None)
parser.add_argument('-o', '--output', help='Output folder. Default for same folder as the source image, 0 for current working folder. Usage: -o .\\output', nargs='?', default=None)
parser.add_argument('-d', '--depth', help='Scan depth. Set 0 for indefinite scan depth', nargs='?', type=int, default=1)
parser.add_argument('-bw', '--borderwidth', help='Border width. Usage: -bw 10 10 10 10', nargs=4, default=[0, 0, 0, 0], type=int)
parser.add_argument('-bc', '--bordercolor', help='RGB border color. Usage: -bc 200 100 100', nargs=3, default=[255, 255, 255], type=int)
# advanced params. don't touch without need
parser.add_argument('-t', '--threshold', help='Object identification threshold (grayscale)', type=int, nargs='?', default=240)
parser.add_argument('--ksize', help='Structure element kernel size', nargs='?', default=11)

parser.add_argument('imgs', help='Input image paths', nargs='*')

args = parser.parse_args()

if args.depth == 0:
    args.depth = None
# endregion

# region [rgba(50,100,100, 0.55)] Progress bar & indicator


class ProgressBar:
    def __init__(self, total: int, prefix='', suffix='', decimals=1, length=100, fill='\u2588', empty='-', end='\r'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.empty = empty
        self.end = end

    def reset(self):
        self.it = 0

    def initialize(self):
        print('\n')
        self.reset()

    def initialize_and_display(self):
        self.initialize()
        self.disp()

    def disp(self):
        percent = ('{0:.' + str(self.decimals) + 'f}').format(100 * (self.it / float(self.total)))
        filled_length = int(self.length * self.it // self.total)
        bar = self.fill * filled_length + self.empty * (self.length - filled_length)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end=self.end)

    def advance(self):
        self.it += 1

    def disp_and_advance(self):
        self.disp()
        self.advance()

    def advance_and_display(self):
        self.advance()
        self.disp()

    def complete(self):
        self.it = self.total

    def complete_and_display(self):
        self.complete()
        self.disp()


class ProgressIndicator:
    def __init__(self, prefix='', symb=['.', '..', '...'], endmsg=''):
        self.prefix = prefix
        self.symb = symb
        self.endmsg = endmsg
        self.it = 0
        self.complete = False

    def advance(self):
        self.it += 1
        if self.it >= len(self.symb):
            self.it = 0

    def disp(self):
        if self.complete:
            print('\r' + self.endmsg)
        else:
            print('\r' + self.prefix + self.symb[self.it], end='')

    def advance_and_display(self):
        self.advance()
        self.disp()

    def disp_and_advance(self):
        self.disp()
        self.advance()

    def set_complete(self):
        self.complete = True

    def complete_and_display(self):
        self.set_complete()
        self.disp()


# endregion

# region [rgba(40,150,200, 0.35)] Directory tree builder (walk)


def walk(top, topDown=True, onError=None, followLinks=False, maxDepth=None):
    """Directory tree generator, similar to os.walk

    Arguments:
        top {string} -- top level of the dir tree

    Keyword Arguments:
        topDown {bool} -- top-down/bottom-up selector. If top-down is selected tuples for parent dirs are returned first (default: {True})
        onError {func} -- error handler (default: {None})
        followLinks {bool} -- follow symbolic links? (default: {False})
        maxDepth {int} -- max tree depth, default value results in behavior identical to os.walk (default: {None})

    Yields:
        (str, list[str], list[str]) -- 3-value tuple (dirpath, dirnames, filenames)
    """
    islink, join, isdir = path.islink, path.join, path.isdir

    try:
        names = os.listdir(top)
    except OSError:
        if onError is not None:
            onError(OSError)
        return

    indicator = ProgressIndicator('Scanning directories', endmsg='\n')

    dirs, nondirs = [], []
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(join(top, name))

    if topDown:
        yield top, dirs, nondirs

    if maxDepth is None or maxDepth > 1:
        for name in dirs:
            indicator.disp_and_advance()
            new_path = join(top, name)
            if followLinks or not islink(new_path):
                for x in walk(new_path, topDown, onError, followLinks, None if maxDepth is None else maxDepth - 1):
                    yield x
    if not topDown:
        yield top, dirs, nondirs
# endregion


print('\nAutomated image cropper.\n(c) Daniil Shuraev, 2020\nBuild %d.%d\nUse -h for help' % (build_major, build_minor))


imgs = args.imgs  # list of images. will be appended if -m|-r|-a are specified

# region [rgba(200,50,100, 0.5)] Image search


def search_dir(regex_string: str):
    """Searches through directories adding images that match specified regex pattern to the processing queue

    Arguments:
        regex_string {str} -- regex pattern to match filenames
    """
    regex = re.compile(regex_string)
    for __, __, files in walk(os.getcwd(), maxDepth = args.depth):
        for fname in files:
            if regex.match(fname):
                imgs.append(fname)


pattern = None

if 'mask' in args:
    pattern = fnmatch.translate(args.mask)
if 'regex' in args:
    if pattern == None:
        pattern = args.regex
    else:
        pattern = '(' + pattern + ')|(' + args.regex + ')'  # append regex to mask if there is any
if args.all:  # match all supported file formats
    pattern = r"\w*.(bmp|pbm|pgm|ppm|sr|ras|jpeg|jpg|jpe|jp2|tiff|tif|png)"

# print(pattern)
if pattern != None:
    search_dir(pattern)
# endregion


ImageFile.LOAD_TRUNCATED_IMAGES = not args.nlti  # load truncated/corrupted files if no -nti arg is passed

bar = ProgressBar(len(imgs), length=50, prefix='Cropping files', suffix='Complete')

bar.initialize_and_display()

for fpath in imgs:

    img = Image.open(fpath)

    if img.mode == 'CMYK':
        img = ImageCms.profileToProfile(img, "USWebCoatedSWOP.icc", "sRGB_Color_Space_Profile.icm", outputMode="RGB")
    # img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

    # (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    __, threshed = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY_INV)

    # (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.ksize, args.ksize))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y + h, x:x + w]

    # add border/padding around the cropped image
    dst = cv2.copyMakeBorder(dst, args.borderwidth[0], args.borderwidth[1], args.borderwidth[2], args.borderwidth[3], cv2.BORDER_CONSTANT, value=args.bordercolor)

    # saving file
    output_path = ''
    ext = pathlib.Path(fpath).suffix  # src file extension
    filename = args.prefix + pathlib.Path(fpath).stem + args.suffix + (ext if args.ext is None else '.' + args.ext)  # [prefix]<filename>[suffix].<extension>

    if args.output == '0':  # current work folder
        output_path = pathlib.Path.cwd()
    elif not args.output is None:
        output_path = args.output
    else:
        output_path = pathlib.Path(fpath).parent

    # make dir if needed
    if not pathlib.Path(output_path).exists():
        try:
            os.makedirs(output_path)
        except OSError:
            print('Failed to create the output directory %s\n\n\n' % output_path)

    output = str(output_path) + '\\' + filename
    cv2.imwrite(output, dst)

    bar.advance_and_display()

print('\n')
