import sys

path = 'data/'
data_types = ['test', 'train', 'valid']

def merge_files(ds_name):
    all_data = ""
    for t in data_types:
        with open(path + ds_name + '/' + t + '.txt', 'r') as f:
            all_data += f.read()
    with open(path + ds_name + '/' + 'triples.txt', 'w') as f:
        f.write(all_data)
    print("All", ds_name, "data merged to triples.txt")


if __name__ == '__main__':
    ds_name = sys.argv[1]
    merge_files(ds_name)