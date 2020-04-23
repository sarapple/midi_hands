import os

class Utils():
    '''utility helpers'''
    @staticmethod
    def get_all_files_in_directory(directory_for_data):
        '''Get the number of files in a directory'''
        all_files = os.listdir(directory_for_data)

        return all_files

    @staticmethod
    def chunks(chunk_source_list, chunk_size):
        '''Yield successive n-sized chunks from lst.'''
        for i in range(0, len(chunk_source_list), chunk_size):
            yield chunk_source_list[i:i + chunk_size]
        
    @staticmethod
    def build_dir_path(file_path, base_folder="output"):
        '''Ensure that directories exist before file is created'''
        full_path = f"./{base_folder}/{file_path}"
        full_path_list = full_path.split("/")
        full_path_list.pop() # remove the filename
        directory_path_without_file = "/".join(full_path_list)

        if not os.path.exists(directory_path_without_file):
            os.makedirs(directory_path_without_file)

        return full_path
