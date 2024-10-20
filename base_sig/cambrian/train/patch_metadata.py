from torch_xla._internal import tpu
import inspect
def _get_metadata_reload(metadata_name):
    import pickle
    def load_metadata(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    env_metadata = load_metadata("/home/wayneyjin/tpu-env.txt") # modify this to your path
    network_endpoint_metadata = load_metadata("/home/wayneyjin/worker-network-endpoints.txt")  # modify this to your path
    if metadata_name == 'tpu-env':
        return env_metadata
    elif metadata_name == 'worker-network-endpoints':
        return network_endpoint_metadata
    else:
        raise ValueError(f"Unknown metadata '{metadata_name}'.")
def update_file_as_str(file_path):
    with open(file_path, 'r') as f:
        file = f.read() # read the file as string
    inject_code = inspect.getsource(_get_metadata_reload)
    metadata_func_str = inspect.getsource(tpu._get_metadata)
    # replace the function name from _get_metadata_reload to _get_metadata
    inject_code = inject_code.replace('_get_metadata_reload', '_get_metadata')
    # replace the content of the function
    file = file.replace(metadata_func_str, inject_code)
    with open(file_path, 'w') as f:
        f.write(file)

if __name__ == "__main__":
    update_file_as_str(tpu.__file__)