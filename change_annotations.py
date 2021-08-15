import xml.etree.ElementTree as ET
import os
import argparse
import sys

parser = argparse.ArgumentParser() 
parser.add_argument("--labels-txt", type=str, default="", help="labels.txt to change the annotation")
parser.add_argument("--new-label", type=str, default="", help="Annotation after the change")
parser.add_argument("--annotations-dir", type=str, default="", help="The directory containing the annotation data you created")
parser.add_argument("--new-annotations-dir", type=str, default="", help="Output directory for rewritten annotations")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

def change_annotations(path, arrow_name, new_arrow_name):
  tree = ET.parse(path)
  root = tree.getroot()
  for child in root.iter('object'):
      for i in list(child):
        if i.tag == 'name' and i.text in arrow_name:
          i.text = new_arrow_name
  return tree

def read_old_annotation_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [i.rstrip() for i in f if i.rstrip() != 'Bull']

if __name__ == '__main__':
    old_labels = read_old_annotation_file(opt.labels_txt)

    for i in os.listdir(opt.annotations_dir):
      read_path = opt.annotations_dir + i
      save_path = opt.new_annotations_dir + i
      tree = change_annotations(read_path, old_labels, opt.new_label)
      tree.write(save_path, encoding='utf-8')

    # new_labels.txt
    new_labels = os.path.join(os.path.dirname(opt.labels_txt), 'new_labels.txt')
    with open(new_labels, 'w', encoding='utf-8') as f:
      f.write('BACKGROUND\n')
      f.write('Bull\n')
      f.write(opt.new_label+'\n')