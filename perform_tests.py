import json
import subprocess
import os
import argparse
from tqdm.auto import tqdm

TIMEOUT = 10*60 # 5 minutes

def launch_c_program(program_path, arglist):
    try:
        result = subprocess.run([program_path] +arglist ,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired as e:
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        raise e
    if result.stderr != "":
        print(f'errors: {result.stderr}')
    return result.stdout, result.stderr


def save_to_json(data, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


def extract_exec_time(output, gpu_only):
    if output is None:
        return {} # timeout
    last_two_lines = output.split('\n')[-3:]

    gpu_id = gpu_only
    gpu_line = last_two_lines[gpu_id]
    assert gpu_line.startswith('gpu')
    gpu_line = gpu_line.split(' ')

    gpu_time_mean = gpu_line[3]
    gpu_time_sd = gpu_line[5][:-1]

    if (not gpu_only):
        cpu_line = last_two_lines[1]
        assert cpu_line.startswith('cpu')
        cpu_line = cpu_line.split(' ')

        cpu_time_mean = cpu_line[3]
        cpu_time_sd = cpu_line[5][:-1]

    res = {"gpu_time_mean":float(gpu_time_mean),
            "gpu_time_sd":float(gpu_time_sd)}

    if (not gpu_only):
        res = res | {"cpu_time_mean":float(cpu_time_mean),
                     "cpu_time_sd":float(cpu_time_sd)}
    
    return res


def partition_output(output, group_dim, iter_num, gpu_only):
    if output is None:
        return {}

    out_splitted = output.split('\n\n')

    global_results = out_splitted[-1]

    group_results = []

    group_offset = (2*iter_num+1) if not gpu_only else (iter_num+1)
    for i in range(0, group_dim*group_offset, group_offset):
        group_res = out_splitted[i:i+group_offset]
        first_iter = group_res[0].split('\n')
        filename = first_iter[0].split(' ')[1]
        
        group_iterations = ['\n'.join(first_iter[2:])] + group_res[1:-1]

        iteration_res = []
        for iteration in group_iterations:
            iter_splitted = iteration.split('\n')
            auction_rounds =  iter_splitted[0].split(' ')[2]
            score =  iter_splitted[1].split(' ')[1]
            exec_time =  iter_splitted[2].split(' ')[3]
            device  = iter_splitted[2].split(' ')[0]
            iteration_res.append({"auction_rounds":int(auction_rounds),
                                  "score":int(float(score)),
                                  "exec_time":float(exec_time),
                                  "device":device})
        
        group_results.append({"filename":filename,
                              "iteration_res":iteration_res})
    
    return {'group_results':group_results,
            'global_results':global_results}


def get_people_num(test_case_fn):
    return extract_test_case_name_info(test_case_fn)['person_number']

def exec_group_and_compute_times(program_path, eps, iter_num, group, gpu_only):
    if eps is None:
        eps = 1.0/(get_people_num(group[0])+1)
    additional_args = [str(eps), str(iter_num)]
    try:
        output,errors = launch_c_program(program_path, additional_args + group)
    except subprocess.TimeoutExpired as e:
        output = e.stdout
        errors = e.stderr
        if output is not None:
            output = output.decode()
        if errors is not None:
            errors = errors.decode()
        raise subprocess.TimeoutExpired({"output":"timeout " +output,"errors":errors },0)
    times = extract_exec_time(output, gpu_only)
    partitioned_output = partition_output(output,len(group), iter_num, gpu_only)
    print(f"group: {group}  , times: {times}")
    return {"times":times, "partitioned_output":partitioned_output, "output":output, "errors":errors}


def extract_test_case_name_info(name, include_test_number = False):
    splitted = name.split('_')
    test_case_number = splitted[1] 
    person_number = splitted[2]
    obj_number = splitted[3]
    density = splitted[4].split('.')[0]

    info = {"person_number":int(person_number),
            "obj_number":int(obj_number),
            "density":float(density)/100} 

    if include_test_number:
        info = info | {"test_case_number":int(test_case_number)}
    
    return info


def extract_group_info(names):
    first_file_info = extract_test_case_name_info(names[0])
    return first_file_info | {"relative_files":names}


def process_files_in_alphabetical_order_per_groups(directory, group_dim, func):
    files = sorted([ os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    results = []
    for i in tqdm(range(0, len(files), group_dim)):
        group = files[i:i+group_dim]
        group_info = extract_group_info(group)
        try:
            results.append(group_info | {"group_values":func(group)})
        except subprocess.TimeoutExpired as e:
            results.append(group_info | {"group_values":e.args[0]})
            return results

    return results


def get_dir_exec_times(dir, same_test_group_dim, exec_path, eps, iter_num, gpu_only):
    extract_group_times = lambda g : exec_group_and_compute_times(exec_path, eps, iter_num , g, gpu_only)
    results = process_files_in_alphabetical_order_per_groups(dir, same_test_group_dim, extract_group_times)

    return {"directory":dir,
            "exec_path":exec_path,
            "iter_num":iter_num,
            "group_dim":same_test_group_dim,
            "directory_results":results}


def main():
    parser = argparse.ArgumentParser(description='Perform time execution tests.')
    parser.add_argument('test_exec', metavar='test_exec', type=str,
                        help='main test executable filename')
    parser.add_argument('group_dim', metavar='gruop_dim', type=int,
                        help='number of consecutive files (in alphabetical order) that refer to the same test case dimension')
    parser.add_argument('iters', metavar='iters', type=int,
                        help='number of iteration for each test case')
    parser.add_argument('test_case_directory', metavar='dir', type=str, nargs='+', # one or more
                        help='directories for the test cases')
    parser.add_argument('--eps', dest='eps', type=float, default=None,
                        help='epsilon value to run the auction, default is 1/(person_num+1) to obtain optimality')
    parser.add_argument('--output_fn', dest='output_filename', default='results.jsom', type=str,
                        help='file to log the performed computation, default results.json')
    parser.add_argument('--gpu_only', action='store_true',
                        help='use when test_exec is compiled with GPUONLY flag')

    args = parser.parse_args()

    assert (args.eps is None or args.eps > 0)

    if (len(args.test_case_directory) != len(set(args.test_case_directory))):
        print("WARNING: duplicates in directory list detected")

    results_filename = args.output_filename

    print(args)

    results = []
    for dir in args.test_case_directory:
        execution = get_dir_exec_times(dir, args.group_dim, args.test_exec, args.eps, args.iters, args.gpu_only)
        results.append(execution)
        print(f"directory {dir} completed, writing current results to {results_filename}")
        save_to_json(results, results_filename) # write periodically to output
    


if __name__ == '__main__':
    main()
