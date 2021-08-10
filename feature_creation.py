import xml.etree.ElementTree as ET
import os
import argparse
import sys
import numpy as np
import math
import csv

parser = argparse.ArgumentParser() 
parser.add_argument("--annotations-dir", type=str, default="", help="The directory containing the annotation data you created")
parser.add_argument("--output-file", type=str, default="", help="Output tsv file")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

def parse_xml(path):
  tree = ET.parse(path)
  root = tree.getroot()
  data_dct = {}
  for child in root.iter('object'):
      for i in list(child):
        # search name tag
        if i.tag == 'name':
          key = i.text
        # search bndbox tag
        if i.tag == 'bndbox':
          data_dct.setdefault(key, [int(j.text) for j in list(i)])

  return data_dct

def calculate_center_coordinate(value):
  center_x = (value[2] + value[0])/2
  center_y = (value[3] + value[1])/2
  return center_x, center_y

def calculate_box_size(value):
  x = value[2] - value[0]
  y = value[3] - value[1]
  return x, y

def calculate_distance(x1, y1, x2, y2):
  return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def calculate_radian(x1, y1, x2, y2):
  return math.atan2(y2-y1, x2-x1)

if __name__ == '__main__':
    annotaion_data = []
    for i in os.listdir(opt.annotations_dir):
        annotaion_data.append(parse_xml(opt.annotations_dir + i))

    # Calculate the position and size of the bbox
    annotaion_data_2 = []
    for i in annotaion_data:
        tmp_dct = {}
        for k, v in i.items():
            tmp_dct.setdefault(k, list(calculate_center_coordinate(v)) + list(calculate_box_size(v)))
        annotaion_data_2.append(tmp_dct)

    # Calculate relative position
    annotaion_data_3 = []
    for i in annotaion_data_2:
        x1, y1 = None, None
        if 'Bull' in i.keys():
            x1 = i['Bull'][0]
            y1 = i['Bull'][1]
        else:
            continue

        tmp_dct = {}
        for k, v in i.items():
            if 'Bull' != k:
                x2 = v[0]
                y2 = v[1]
                dis = calculate_distance(x1, y1, x2, y2)
                rad = calculate_radian(x1, y1, x2, y2)
                tmp_dct.setdefault(k, [dis, rad, v[2], v[3]])
        annotaion_data_3.append(tmp_dct)

    dct_coordinates = {}
    for i in annotaion_data_3:
        for k, v in i.items():
            dct_coordinates.setdefault(k, [])
            dct_coordinates[k].append(v)

    with open(opt.output_file, 'w', newline='', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for k, v in dct_coordinates.items():
            for i in v:
              tsv_writer.writerows([[k, str(i[0]), str(i[1]), str(i[2]), str(i[3])]])