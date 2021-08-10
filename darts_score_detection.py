import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np
import math
import onnxruntime

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--model", type=str, default="", help="")
parser.add_argument("--dnn-model", type=str, default="", help="")
parser.add_argument("--labels", type=str, default="", help="")
parser.add_argument("--input-blob", type=str, default="", help="")
parser.add_argument("--output-cvg", type=str, default="", help="")
parser.add_argument("--output-bbox", type=str, default="", help="")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

info = jetson.utils.cudaFont()

class DartsScoreDetection(object):
	def __init__(self, score_range=None, dnn_model=None):
		self.points = [11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 0]
		self.score_map = {}
		self.index_to_lable = {0:'Double', 1:'Single', 2:'Triple'}
		self.score_range = score_range
		self.dnn_model = dnn_model

	def create_score_map(self, score_range):
		for i, _ in enumerate(self.points):
			if i == 10:
				self.score_map[i] = (score_range[i], np.pi)
			else:
				self.score_map[i] = (score_range[i], score_range[i+1])

		self.points.append(11)
		self.score_map[20] = (-np.pi, score_range[11])

	def calculate_score(self, x1, y1, x2, y2, box_width, box_height):
		distance = self.calculate_distance(x1, y1, x2, y2)
		rad = self.calculate_radian(x1, y1, x2, y2)
		multiple = ''

		ort_inputs = {self.dnn_model.get_inputs()[0].name: [[distance, rad, box_width, box_height]]}
		multiple = np.argmax(self.dnn_model.run(None, ort_inputs)[0])
		
		res = self.binary_search(self.score_range, rad)
		predict_score = self.points[res]

		return self.index_to_lable[multiple] + ' ' + str(predict_score)

	def calculate_distance(self, x1, y1, x2, y2):
		return np.sqrt((x2-x1)**2 + (y2-y1)**2)

	def calculate_radian(self, x1, y1, x2, y2):
		return math.atan2(y2-y1, x2-x1)

	def binary_search(self, numbers, value):
		def _binary_search(numbers, value, left, right):
			if left > right:
				return 21

			mid = (left + right) // 2
			if numbers[mid] <= value and value < numbers[mid+1]:
				return mid
			elif numbers[mid] < value:
				return _binary_search(numbers, value, mid+1, right)
			else:
				return _binary_search(numbers, value, left, mid-1)

		return _binary_search(numbers, value, 0, len(numbers)-1)


if __name__ == '__main__':
	unit = np.pi/10
	scale = -unit/2
	radians = [scale]
	for i in range(10):
		scale+=unit
		radians.append(scale)
	radians.append(np.pi)
	score_range = sorted([i * -1 for i in radians][2:]) + sorted(radians)

	dsd = DartsScoreDetection(
				score_range = score_range,
				dnn_model = onnxruntime.InferenceSession(opt.dnn_model)
	)

	center_x, center_y = 0, 0

	while output.IsStreaming():
		img = input.Capture()
		detections = net.Detect(img)
		for detection in detections:
			if detection.ClassID == 1:
				center_x = detection.Center[0]
				center_y = detection.Center[1]
			else:
				info.OverlayText(img, 5, 5,"score:{0}".format(
					dsd.calculate_score(center_x, center_y, detection.Center[0], detection.Center[1], detection.Width, detection.Height)),
					int(detection.Left)+5,
					int(detection.Top)+35,
					info.White, 
					info.Gray40
					)

			output.Render(img)
			output.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

