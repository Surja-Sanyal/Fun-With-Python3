#Program:   Implementation of Irene Lo RLDA Mechanism
#Inputs:    Number of Schools
#           Number of Students
#           Students' First Round Preferences
#           Schools' Preferences
#           Students' Second Round Preferences
#Outputs:   Students' Lottery Numbers
#           Students' First Round Allocations
#           Students' Second Round Allocations
#Author:    Surja Sanyal
#Date:      24 JUN 2020
#Comments:  None




##   Start of Code   ##


#   Imports    #
import os
import re
import sys
import math
import copy
import time
import psutil
import shutil
import random
import resource
import datetime
import traceback
import itertools
import numpy as np
import multiprocessing
from textwrap import wrap
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats import truncnorm



##  Global environment   ##

#   Customize here  #
INFINITY            = 1000
DECIMAL_PRECISION   = 23                    # = Float32 mantissa
TESTDATA_LOAD       = ""                    # = "TestData"
TESTDATA_STORE      = ""                    # = "TestData"
ALPHA               = 9                     # = Integers in interval [-9, 9]


#   Do not change   #
START_ALPHA         =  -9
END_ALPHA           =   9
NUMERIC_DELIMITER   = -99
LOCK                = multiprocessing.Lock()
CPU_COUNT           = multiprocessing.cpu_count()
MEMORY              = math.ceil(psutil.virtual_memory().total/(1024.**3))
FACTOR              = CPU_COUNT * int(2 ** 12 / (CPU_COUNT ** 3 * MEMORY))
DATA_LOAD_LOCATION  = os.path.dirname(sys.argv[0]) + "/" + TESTDATA_LOAD
DATA_STORE_LOCATION = os.path.dirname(sys.argv[0]) + "/" + TESTDATA_STORE 



##  Function definitions    ##

#   Load data option    #
def get_data_generation_options():

    print_locked("\nDo you want to auto-generate student and school data (Y/N): ", end="")
    data_auto_generate = input()
    
    return data_auto_generate



#   Get number of executions    #
def get_num_executions():

    print_locked("\nHow many times do you want Round II to execute?: ", end="")
    num_executions = input()

    try:
        num_executions = int(num_executions)
    
    except Exception:
    
        num_executions = 1
        print_locked("\nExecutions defaulted to only once.")

    return num_executions



#   Load student preferences    #
def load_student_preferences(students):

    lottery = []

    #   Read student preferences    #
    with open(DATA_LOAD_LOCATION + "_Student_data.txt", "r") as fp:
    
        print ("")
        
        for serial, line in enumerate(fp):
        
            print ("Loading data for Student number:", serial + 1, end="\r")
            per_student = per_student_line(line.strip())
            lottery.append(per_student[-2])
            students.append(per_student)
        
        print ("\n\nStudent data load complete.")
    
    return lottery, students



#   Load preferences per student    #
def per_student_line(line):

    student = []
    
    parts=[j for j in re.split("\], \[", line)]
    
    for piece in parts:
    
        student.extend([[convert(k) 
        for k in filter(None, re.split("\[|, |\]", piece))]])

    return student



#   Load school preferences    #
def load_school_preferences(schools, FLDA_or_RLDA = "", folder_location = ""):

    schools = []
    
    #   Check for FLDA or RLDA data     #
    if (len(FLDA_or_RLDA) != 0):
    
        FLDA_or_RLDA = "_" + FLDA_or_RLDA
    
    #   Read school preferences     #
    with open(DATA_LOAD_LOCATION + folder_location + FLDA_or_RLDA + "_School_data.txt", "r") as fp:
    
        print ("")
        
        for serial, line in enumerate(fp):
        
            print ("Loading data for School number:", serial + 1, end="\r")
            schools.append([convert(i) for i in filter(None, re.split("\[|, |\]", line.strip()))])
    
        print ("\n\nSchool data load complete.")
        
    return schools



#   Convert string to int or float    #
def convert(some_value):

    try:
        return int(some_value)
        
    except ValueError:
    
        try:
        
            return float(some_value)
        
        except ValueError:
        
            print_locked(traceback.format_exc())



#   Take number of students and schools as inputs   #
def input_numbers():

    try:
        print_locked("\nPlease enter the number of students: ", end="")
        student_count = int(input())
        print_locked("\nPlease enter the number of schools: ", end="")
        school_count = int(input())
        
    except Exception:
    
        student_count, school_count = -1, -1
        print_locked("\nStudent and School data defaulted to those in the last execution.")

    return student_count, school_count



#   Directly start from Round II option  #
def check_round_to_start_with():

    try:
    
        print_locked("\nStart from Round I or Round II? (1/2): ", end="")
        round_select = int(input())
        
        if round_select not in (1, 2):
        
            raise Exception
        
    except Exception:
    
        round_select = 2
        print_locked("\nRound selection option defaulted to Round II.")

    return round_select



#   Generate preferences per school     #
def generate_per_school(index, student_count, school_count):

    return [random.choices(range(1, 50), weights = range(1, 50))[0], 
        random.randint(1, max(2, int(student_count/(school_count * 2))))]



#   Generate student data   #
def generate_student_data(student_count, school_count, 
                          cpu_count, preset_factor, schools, queue):
    
    pool = multiprocessing.Pool(cpu_count)
    
    #   Lottery generation  #
    start = datetime.datetime.now()
    
    lottery, lottery_round_2 = pool.map(partial(get_truncated_normal, student_count = student_count), range(2))
   
    lottery_generation_time = datetime.datetime.now() - start
    queue.put(lottery_generation_time)
    queue.put(lottery)
    
    print_locked("\nLottery generation time =", 
    lottery_generation_time, "hours")
    
    #   Generate students in chunks    #
    if (student_count < preset_factor ** 2):
    
        factor = 1
        chunk_size = student_count
        student_chunk = [None]
    
    else:
        factor = preset_factor
        chunk_size = math.ceil(student_count / factor)
        student_chunk = [None] * cpu_count
        print_locked("\nLarge data detected with Big-Oh(", student_count * school_count, 
            ") processing complexity..", sep="")
        
    print_locked("\nChunk size set to", chunk_size, "records..")
    print_locked("\nTotal", factor, "chunks of data to be processed..")
        
    for i in range(factor):
    
        start = datetime.datetime.now()
    
        start_loc = i * chunk_size
        
        if (i == factor - 1):
        
            chunk_size = min(chunk_size, student_count - ((factor - 1) * chunk_size))
        
        end_loc = start_loc + chunk_size
        
        #   School student preferences   #
        school_preferences_per_students = \
            pool.imap_unordered(partial(create_school_preferences_per_student, 
            schools = schools), ([school_count] * chunk_size),
            chunksize = math.ceil(chunk_size/cpu_count))
        
        #   Student preferences   #
        student_preferences = \
            pool.imap_unordered(partial(preference_per_student, 
            school_count = school_count), range(chunk_size),
            chunksize = math.ceil(chunk_size/cpu_count))
        
        #   Assemble students   #
        student_chunk[i % cpu_count] = \
            pool.starmap(get_students, 
            zip(student_preferences, school_preferences_per_students, 
            lottery[start_loc:end_loc], lottery_round_2[start_loc:end_loc]),
            chunksize = math.ceil(chunk_size/cpu_count))
        
        student_preferences = None
        school_preferences_per_students = None
        
        student_generation_time = datetime.datetime.now() - start
        print_locked("\nStudent chunk", i +  1, "out of", factor, 
        "generation time =", student_generation_time, "hours")
        
        #   Get student data directly to RAM  #
        if (factor == 1):
            
            print_locked("\nSaving student data..")
            save_start = datetime.datetime.now()
            
            write_partial_data_to_file(0, student_chunk[0])
            student_chunk = [None]
            
            print_locked("\nData save time =", 
            datetime.datetime.now() - save_start, "hours")
        
        #   Store partial student data to temporary files   #
        elif (i % cpu_count == cpu_count - 1):
        
            print_locked("\nSaving last", cpu_count, "chunks of student data..")
            save_start = datetime.datetime.now()
            
            pool.starmap(write_partial_data_to_file, zip(range(cpu_count), student_chunk))
            student_chunk = [None] * cpu_count
            
            print_locked("\nData save time =", 
            datetime.datetime.now() - save_start, "hours")



#   Get complete students   #
def get_students(student, school_preference, lottery, lottery_round_2):

    return [student] + [school_preference] + [lottery] + [lottery_round_2]

    

#   Generate preferences per student    #
def preference_per_student(index, school_count):

    #   Weighted preferences     #
    random_preference = \
        list(np.random.choice(range(1, school_count + 1), 
        random.choices(range(school_count + 1), weights = range(1, school_count + 2))[0],
        replace = False,
        p = [(val/sum(range(school_count + 1))) for val in reversed(range(1, school_count + 1))]))

    #   Put data into students list     #
    return random_preference + [school_count + 1]



#   Add preference to each student  #
def create_school_preferences_per_student(school_count, schools): #

    return [random.choices(range(-1, schools[i][0] - 1), 
            weights = range(1, schools[i][0] + 1))[0] for i in range(school_count)]



#   Get a normal distribution   #
def get_truncated_normal(index, student_count = 0, mean=0, sd=1, low=0, upp=1, truncate = 0):

    np.random.seed(index)

    if (truncate == 0):
    
        return [[item] for item in np.random.normal(loc = mean, scale = sd, size = student_count)]
    
    else:
    
        return [[item] for item in truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(student_count)]


#   Lottery generation      #
def lottery_generation(student_count):
    
    return [np.random.choice([[round(item/student_count, len(str(student_count)))] \
        for item in range(student_count)], student_count)]



#   Write data to file  #
#   Students:   
#   [Student_Pref]    |   [Schools_Prefs]   |   [Lottery_No.]   #
#   Schools:    
#   [School_Pref_Groups]    |   [School_Vacancies]  #
def write_data_to_file(list_name, list_data, new_folder = ""):

    #   Write data  #
    with open(DATA_STORE_LOCATION + new_folder + "_" + list_name + "_data.txt", "w") as fp:
        
        [fp.write(str(line) + "\n") for line in list_data]



#   Write partial student data to file  #
def write_partial_data_to_file(chunk_number, chunk_data):

    #   Write data  #
    with open(DATA_STORE_LOCATION + ".TempData/" + str(chunk_number) + "_Student_data.txt", "a") as fp:
    
        [fp.write(str(line) + "\n") for line in chunk_data]



#   Write execution results to file     #
def write_execution_results(execution_number, student_count, school_count, numeric_delimiter, data_to_write):

    #   Write data  #
    with open(DATA_STORE_LOCATION + "_Execution_results.txt", "a") as fp:
    
        for each_execution in data_to_write:
        
            fp.write(str(execution_number) + "\t" + str(student_count) + "\t" + str(school_count) 
                + "\t" + str(each_execution[0]) + "\t" + str(each_execution[1]) + "\n")



#   Display execution results   #
def display_execution_results(data_store_location, num_executions, numeric_delimiter):

    #   Print Headers   #
    print_locked('\n\n{:.{align}{width}}'.format("Execution Results", align='^', width=60))
    
    print_locked("\nExecution Results: Relocations by FLDA and RLDA")
    
    print_locked('\n{:.{align}{width}}'.format("Relocation Plot Dataset", align='^', width=60), end="\n\n")
    
    print_locked(
        '{:{align}{width}}'.format("[Exec. no.]", 
            align='>', width=len("[Exec. no.]")),
        '{:.{align}{width}}'.format("[Students]", 
            align='<', width=len("[Students]")),
        '{:.{align}{width}}'.format("[Schools]", 
            align='<', width=len("[Schools]")),
        '{:.{align}{width}}'.format("[Alpha(α)]", 
            align='<', width=len("[Alpha(α)]")),
        '{:{align}{width}}'.format("[Relocations]", 
            align='<', width=len("[Relocations]")),
        "\n")
    
    
    #   Set up variables for plotting   #
    execution, students, schools, raw_alpha, raw_RLDA, ideal_PLDA, data = [], [], [], [], [], [], []
    infinity = INFINITY
    
    #   Read the file for data  #
    with open(data_store_location + "_Execution_results.txt", "r") as fp:
        
        for line in fp:
        
            data[:] = line.strip().split()
        
            #   Save data for the graph     #
            execution.append(int(data[0]) if int(data[3]) in (-infinity, infinity) else None)
            students.append(int(data[1]) if int(data[3]) in (-infinity, infinity) else None)
            schools.append(int(data[2]) if int(data[3]) in (-infinity, infinity) else None)
            ideal_PLDA.append(int(data[4]) if int(data[3]) in (-infinity, infinity) else None)
            raw_alpha.append(int(data[3]))
            raw_RLDA.append(int(data[4]) if int(data[3]) not in (-infinity, infinity) else int(data[3]))
        
            #   Display data    #
            print_locked(
                '{:{align}{width}}'.format(data[0] + ":", 
                    align='>', width=len("[Exec. no.]")),
                '{:.{align}{width}}'.format(data[1], 
                    align='<', width=len("[Students]")),
                '{:.{align}{width}}'.format(data[2], 
                    align='<', width=len("[Schools]")),
                '{:.{align}{width}}'.format(data[3] 
                    if int(data[3]) not in (-infinity, infinity) 
                    else ("RLDA(-∞)" if int(data[3]) == -infinity 
                                     else "FLDA(+∞)"), 
                    align='>', width=len("[Alpha(α)]")),
                '{:.{align}{width}}'.format(data[4], 
                    align='>', width=len("[Relocations]")),
                "\n")

    print_locked('\n{:.{align}{width}}'.format("End of Results", align='^', width=60))

    #   Display execution data in a graph   #
    display_graph(data_store_location, execution, students, schools, raw_alpha, raw_RLDA, ideal_PLDA, 1)



#   Display execution data in a graph   #
def display_graph(data_store_location, execution = [], students = [], 
                    schools = [], raw_alpha = [], raw_RLDA = [], ideal_PLDA = [], show_plot = 0):

    #   Read data   #
    infinity = INFINITY
    
    if (len(raw_alpha) == 0):
    
        #   Read the file for data  #
        data = []
        
        with open(data_store_location + "_Execution_results.txt", "r") as fp:
            
            for line in fp:
            
                data[:] = line.strip().split()
                
                #   Save data for the graph     #
                execution.append(int(data[0]) if int(data[3]) in (-infinity, infinity) else None)
                students.append(int(data[1]) if int(data[3]) in (-infinity, infinity) else None)
                schools.append(int(data[2]) if int(data[3]) in (-infinity, infinity) else None)
                ideal_PLDA.append(int(data[4]) if int(data[3]) in (-infinity, infinity) else None)
                raw_alpha.append(int(data[3]))
                raw_RLDA.append(int(data[4]) if int(data[3]) not in (-infinity, infinity) else int(data[3]))

    
    #   Get min and max values  #
    min_alpha = min(filter(lambda x: x not in (-infinity, infinity), raw_alpha))
    max_alpha = max(filter(lambda x: x not in (-infinity, infinity), raw_alpha))
    
    #   Split lists on delimiter value     #
    execution[:] = filter(lambda x: x is not None, execution)
    students[:] = filter(lambda x: x is not None, students)
    schools[:] = filter(lambda x: x is not None, schools)
    ideal_PLDA[:] = filter(lambda x: x is not None, ideal_PLDA)
    min_ideal_PLDA, max_ideal_PLDA = ideal_PLDA[-1], ideal_PLDA[-2]
    
    alpha = [list(y) for x, y in itertools.groupby(raw_alpha, lambda z: z in (-infinity, infinity)) if not x]
    RLDA = [list(y) for x, y in itertools.groupby(raw_RLDA, lambda z: z in (-infinity, infinity)) if not x]

    #   Create the figure   #
    comparision = plt.figure("Figure: Number of reassigned students versus α.", figsize=(12, 10))
    
    if (not isinstance(RLDA[0], int)):
    
        [plt.plot(alpha[i], RLDA[i], 
            label="Plot: " + str(i + 1) + " | Students: " + str(students[i]) 
            + " | Schools: " + str(schools[i]) + " | Exec.: " + str(execution[i]) 
            + " | α ∈ [" + str(min(alpha[i])) + ", " + str(max(alpha[i])) + "]", 
            marker='o') 
            for i in range(len(RLDA))]
        
    else:
        
        plt.plot(alpha, RLDA,
            label="Plot: 1 | Students: " + str(students[0]) 
            + " | Schools: " + str(schools[0]) + " | Exec.: " + str(execution[0]) 
            + " | α ∈ [" + str(min(alpha)) + ", " + str(max(alpha)) + "]", 
            color='k', marker='o')

    #   Customize plot   #
    plt.xlabel('Alpha (α) -->')
    plt.ylabel('Relocations -->')
    
    plt.axhline(y = max_ideal_PLDA, color = "k", linestyle = ":")
    plt.axhline(y = min_ideal_PLDA, color = "k", linestyle = "--")
    
    plt.annotate('FLDA', xy =(max_alpha + 0.1, max_ideal_PLDA + (max_ideal_PLDA - min_ideal_PLDA) * 0.01))
    plt.annotate('RLDA', xy =(min_alpha - 0.9, min_ideal_PLDA - (max_ideal_PLDA - min_ideal_PLDA) * 0.03))
    
    plt.xticks(np.arange(min_alpha - 1, max_alpha + 1, step = 1))
#    plt.yticks(np.arange(0, 8000, step = 1000))
    
    plt.legend(loc="center left", facecolor="white", framealpha=1)
    
    plt.title("\n".join(wrap("Figure: Number of reassigned students versus α. " 
        + "The number of reassigned students under the extreme values of α, " 
        + "namely, α = ∞ (FLDA) and α = −∞ (RLDA), are shown via dotted lines.", 
        width=10 * 10)), 
        y=1.05, loc = "left")
    
    comparision.savefig(data_store_location + "Figure: Number of reassigned students versus α.pdf", bbox_inches='tight')
    comparision.savefig(data_store_location + "Figure: Number of reassigned students versus α.jpg", bbox_inches='tight')
    
    if (show_plot == 1):
    
        plt.show()



#   Load partial student data   #
def get_partial_student_data(part_number, total_parts):

    start = datetime.datetime.now()
    student_part = []

    #   Read student preferences    #
    with open(DATA_LOAD_LOCATION + ".TempData/" + str(part_number) + "_Student_data.txt", "r") as fp:

        for line in fp:
            
            student_part.append(per_student_line(line.strip()))
    
    print_locked("\nPart", part_number + 1, 
        "out of", total_parts, "load time:", 
        datetime.datetime.now() - start, "hours")
    
    return student_part



#   Get index of an element in a list    #
def get_index(schools, lottery_number):

    for i, school in enumerate(schools):
        
        if (lottery_number in school[3:]):
        
            return i, school[3:].index(lottery_number) + 3
            
    return -1, -1



#   Custom print lists    #
def custom_print(array_name, array_data):

    print_locked("\n" + array_name + ":")
    
    #   Calculate column widths     #
    if (array_name == "Students"):
        
        #   Get columns     #
        transpose_array_data = list(zip(*array_data))
    
        #   Student column widths     #
        stud_col_1=len(str(len(array_data)))
        stud_col_2=max(max([len(str(j)) for j in transpose_array_data[0]]), 
            len("[Student_Preferences]"))
        stud_col_3=max(max([len(str(j)) for j in transpose_array_data[1]]), 
            len("[Schools'_Preference_Groups]"))
        stud_col_4=max(max([len(str(j)) for j in transpose_array_data[2]]), 
            len("[Round_I_Lottery_Number]"))
        stud_col_5=max(max([len(str(j)) for j in transpose_array_data[3]]), 
            len("[Round_II_Lottery_Number]"))
    
    #   School column widths  #
    schl_col_1=max(len(str(len(array_data))), len("[Sl]"))
    schl_col_2=len("[Pref_Groups]")
    schl_col_3=len("[Vacancies]")
    schl_col_4=len("[School_Cutoff]")
    schl_col_5=len("[Allocations-->]")
    
    for i in range(len(array_data)):

        if (array_name == "Students"):
        
            #   Print columns  #
            if (i==0):
                print_locked(
                    '{:{align}{width}}'.format("[Sl]", 
                        align='>', width=stud_col_1 + 2),
                    '{:.{align}{width}}'.format("[Student_Preferences]", 
                        align='<', width=stud_col_2),
                    '{:.{align}{width}}'.format("[Schools'_Preference_Groups]", 
                        align='<', width=stud_col_3 + 2),
                    '{:.{align}{width}}'.format("[Round_I_Lottery_Number]", 
                        align='<', width=stud_col_4),
                    '{:{align}{width}}'.format("[Round_II_Lottery_Number]", 
                        align='<', width=stud_col_5),
                    "\n")
            
            #   Print data  #
            print_locked(
                '{:{align}{width}}'.format(str(i + 1) +":", 
                    align='>', width=stud_col_1 + 2),
                '{:.{align}{width}}'.format(str(array_data[i][0]), 
                    align='<', width=stud_col_2),
                '{:.{align}{width}}'.format(str(array_data[i][1]), 
                    align='<', width=stud_col_3 + 2),
                '{:.{align}{width}}'.format(str(array_data[i][2]), 
                    align='<', width=stud_col_4),
                '{:{align}{width}}'.format(str(array_data[i][3]), 
                    align='<', width=stud_col_5),
                )

        else:
        
            #   Print columns  #
            if (i==0):
                print_locked(
                    '{:{align}{width}}'.format("[Sl]", 
                        align='>', width=schl_col_1 + 2),
                    '{:.{align}{width}}'.format("[Pref_Groups]", 
                        align='<', width=schl_col_2 + 2),
                    '{:.{align}{width}}'.format("[Vacancies]", 
                        align='<', width=schl_col_3 + 2),
                    '{:.{align}{width}}'.format("[School_Cutoff]", 
                        align='<', width=schl_col_4 + 2),
                    '{:{align}{width}}'.format("[Allocations-->]", 
                        align='<', width=schl_col_5),
                    "\n")
            
            #   Print data  #
            print_locked(
                '{:{align}{width}}'.format(str(i + 1) +":", 
                    align='>', width=schl_col_1 + 2),
                '{:.{align}{width}}'.format(str(array_data[i][0]), 
                    align='^', width=schl_col_2 + 2),
                '{:.{align}{width}}'.format(str(array_data[i][1]), 
                    align='^', width=schl_col_3 + 2),
                '{:.{align}{width}}'.format(str(array_data[i][2]), 
                    align='^', width=schl_col_4 + 2),
                '{:{align}{width}}'.format(str(array_data[i][3:]), 
                    align='<', width=schl_col_5)
                )



#   Print with lock    #
def print_locked(*content, sep=" ", end="\n"):

    with open(DATA_STORE_LOCATION + "_Log_File.txt", "a") as log_file:
    
        try:
        
            with lock:
            
                print (*content, sep = sep, end = end)
                print (*content, sep = sep, end = end, file=log_file)

        except Exception:
        
            with LOCK:
            
                print (*content, sep = sep, end = end)
                print (*content, sep = sep, end = end, file=log_file)



#   Print timer data    #
def print_information(data_auto_generate, check_round, information, student_count, 
                      school_count, num_executions, current_execution, infinity):
    
    index = 0
    
    #   Print general details   #
    print_locked('\n\n\n{:.{align}{width}}'.format("Last Run Statistics", align='^', width=90))
    
    print_locked("\nStatistics for last run:")
    
    print_locked('\n{:.{align}{width}}'.format("Round I", align='^', width=90))
    
    print_locked("\nNumber of students participating\t=", student_count)
    
    print_locked("\nNumber of schools participating\t\t=", school_count)
    
    print_locked("\nTotal number of Round II executions\t=", num_executions)
    
    print_locked("\nNumber of executions completed\t\t=", current_execution)
    
    print_locked("\nProgram CPU oriented start time\t\t=", information[index], "hours")
    index += 1
    
    if data_auto_generate.upper() == 'Y':
    
        print_locked("\nSchool generation time\t\t\t=", information[index], "hours")
        index += 1
        
        print_locked("\nLottery generation time\t\t\t=", information[index], "hours")
        index += 1
        
        print_locked("\nStudent generation time\t\t\t=", information[index], "hours")
        index += 1
    
    else:
    
        print_locked("\nData load time\t\t\t\t=", information[index], "hours")
        index += 3
    
    #   Print Round I details   #
    if (check_round == 1):
        print_locked("\nCutoff generation time\t\t\t=", information[index], "hours")
        index += 1
        
        print_locked("\nInitial DA-STB took\t\t\t=", information[index], "hours")
        index += 1
    
    else:
    
        index += 2
    
    #   Print Round II details   #
    if (num_executions > 0):
    
        #   RLDA    #
        print_locked('\n\n{:.{align}{width}}'.format("Round II PLDA", align='^', width=90))
        
        print_locked("\nTotal RLDA DA-STB took\t\t\t=", information[-3], "hours")
        
        print_locked("\nStudents relocated by RLDA\t\t= ", end="")
        
        print_locked('{:{align}{width}}'.format("[Alpha(α)]", align='>', width=len("[Alpha(α)]")),
            '{:{align}{width}}'.format("[Relocations]", align='>', width=len("[Relocations]")),
            '{:{align}{width}}'.format("[Duration (hours)]", align='>', width=len("[Duration (hours)]")))
               
        [print_locked("\t\t\t\t\t", '{:{align}{width}}'.format(str(alpha) 
            if alpha not in (-infinity, infinity)
            else ("FLDA(+∞)" if alpha == infinity else "RLDA(-∞)"), 
            align='>', width=len("[Alpha(α)]")),
            '{:.{align}{width}}'.format(str(relocations), 
            align='>', width=len("[Relocations]")),
            '{:.{align}{width}}'.format(str(duration), 
            align='>', width=len("[Duration (hours)]")))
               
        for alpha, relocations, duration in information[-4]]
        
        #   Print exection time details   #
        print_locked('\n\n{:.{align}{width}}'.format("Total", align='^', width=90))
        
        print_locked("\nTotal program CPU execution time\t=", information[-2], "hours")

    print_locked('\n{:.{align}{width}}'.format("End of Statistics", align='^', width=90), end="\n\n")



#   Calculate first round cutoffs   #
def calculate_cutoffs(students, schools):

    #   Calculate decimal precision    #
    decimal_precision = DECIMAL_PRECISION
    interested_students = []
    
    #   Calculate cutoffs   #
    for k in range(len(schools)):
    
        #   Enlist interested students     #
        interested_students[:] = [interested_student 
            for interested_student in students 
            if (k + 1) in interested_student[0]]
        
        for i, student in enumerate(sorted(interested_students, 
            key=lambda z: (-(z[:][1][k] + z[:][2][0])))):
        
            if (i + 1 == schools[k][1] or i + 1 == len(interested_students)):
            
                schools[k].extend([round(student[1][k] 
                    + student[2][0], decimal_precision)])
                break



#   Invoke first round DA-STB as required   #
def invoke_assignment(students, schools, cpu_count, numeric_delimiter):

    #   Calculate decimal precision    #
    decimal_precision = DECIMAL_PRECISION
    
    #   DA-STB until seats fill     #
    cutoff_adjusted=0
    rerun_assignment=1
    vacant_schools, adjusted_schools = [], []
    
    da_1_start = datetime.datetime.now()
    print_locked("\nInitial DA-STB start")
    
    while (rerun_assignment==1):
        
        #   Run DA_STB    #
        schools = DA_STB(students, schools, cpu_count)
        
        #   Check for empty seats in schools    #
        vacant_schools = [i for i, school in enumerate(schools) if len(school[3:]) < school[1]]
        
        if (len(vacant_schools) > 0):
            
            #   Adjust cutoff for the school    #
            adjusted_schools = \
                pool.starmap(partial(adjust_cutoff, 
                students = [interested_student for interested_student in students 
                if (get_index(schools, interested_student[2][0])[0] < 0)], 
                decimal_precision = decimal_precision,
                numeric_delimiter = numeric_delimiter), 
                zip(vacant_schools, [schools[i] for i in vacant_schools]), 
                chunksize = math.ceil(len(vacant_schools)/cpu_count))
            
            schools = [adjusted_schools[vacant_schools.index(i)] 
                if i in vacant_schools else school 
                for i, school in enumerate(schools)]
            
            cutoff_adjusted=1
        
        #   Check rerun condition   #
        if (cutoff_adjusted > 0):
        
            rerun_assignment=1
            cutoff_adjusted=0
            
        else:
        
            rerun_assignment=0

    time_taken = datetime.datetime.now() - da_1_start
    print_locked("\nInitial DA-STB took", time_taken, "hours")
    
    return time_taken, schools



#   First round DA-STB  #
def DA_STB(students, schools, cpu_count):

    #   Calculate decimal precision    #
    decimal_precision = DECIMAL_PRECISION
    
    #   Allocate per school    #
    for i in range(len(schools)):
    
        #   Enlist interested students     #
        interested_students = [interested_student 
            for interested_student in students 
            if (i < len(interested_student[0]) - 1
            and get_index(schools, interested_student[2][0])[0] < 0)]
        
        schools = pool.starmap(partial(DA_STB_per_school, 
            students = interested_students, school_count = len(schools),
            decimal_precision = decimal_precision, i = i), 
            enumerate(schools), chunksize = math.ceil(len(schools)/cpu_count))
        
    return schools



#   Parallely allocate to schools   #
def DA_STB_per_school(k, school, students, school_count, decimal_precision, i):
    
    interested_students = [interested_student 
        for interested_student in students 
        if (interested_student[0][i] == k + 1)]

    for j, student in enumerate(sorted(interested_students, 
        key=lambda z: (-(z[:][1][k] + z[:][2][0])))):
    
        #   Break if school seats are full  #
        if (len(school[3:]) == school[1]):
            break
            
        #   Get private school position    #
        private_school_position=student[0].index(school_count + 1)
        
        #   Give proposals    #
        if (i < private_school_position):
            
            if (round(student[1][k] + student[2][0], 
                decimal_precision) >= school[2]):
            
                school.extend([student[2][0]])

    return school



#   Adjust first round cutoff for the school    #
def adjust_cutoff(index, school, students, decimal_precision, numeric_delimiter):

    #   Initialize current cutoff index     #
    current_cutoff_index=numeric_delimiter
    
    #   Enlist interested students     #
    interested_students = [interested_student 
        for interested_student in students 
        if ((index + 1) in interested_student[0])]
    
    current_cutoff_index = -1
    
    #   Loop by students to get required cutoff     #
    for i, student in enumerate(sorted(interested_students, 
        key=lambda z: (-(z[:][1][index] + z[:][2][0])))):
    
        #   Get current cutoff position     #
        if (round(student[1][index] + student[2][0], 
            decimal_precision) <= school[2]
            and current_cutoff_index < 0):
            
            current_cutoff_index=i
        
        #   Break loop if no reduction in cutoff can fill school seats     #
        if (i == len(students) - 1): 
        
            school[2]=round(student[1][index] 
            + student[2][0], decimal_precision)
            break
        
        #   Reduce cutoff to fill in school seats   #
        if (i == current_cutoff_index 
        + max((school[1] - len(school[3:])), 1)
        and current_cutoff_index > -1):
            
            school[2]=round(student[1][index] 
            + student[2][0], decimal_precision)
            break

    return school



#   Remove 10% students as is the trend for dropouts    #
def dropouts(school, dropout_lotteries, numeric_delimiter):
    
    return school[:3] + [numeric_delimiter if x in dropout_lotteries else x for x in school[3:]]



#   Update priority of allocated student for Round II    #
def update_priorities(student, schools):

    school_number, position_number = get_index(schools, student[2][0])
    
    if (school_number > -1):
    
        student[1][school_number] = schools[school_number][0]

    return student



#   Calculate second round cutoffs   #
def calculate_cutoffs_RLDA(students, schools, dropout_lotteries, numeric_delimiter, alpha):

    #   Calculate decimal precision    #
    decimal_precision = DECIMAL_PRECISION
    schools_got_vacancy, interested_students = [], []
    
    #   Calculate cutoffs   #
    schools_got_vacancy[:] = [i for i in range(len(schools)) 
        if (numeric_delimiter in schools[i])]
    
    for k in schools_got_vacancy:
    
        #   Enlist unallocated students     #
        interested_students[:] = [interested_student 
            for interested_student in students 
            if ((k + 1) in interested_student[0]
            and interested_student[2][0] not in dropout_lotteries)]
    
        for i, student in enumerate(sorted(interested_students, 
            key=lambda z: (-(z[:][1][k] + (z[:][3][0] + alpha * z[:][2][0]))))):
        
            if (i + 1 == schools[k].count(numeric_delimiter + schools[k][1]) 
                or i + 1 == len(interested_students)):
            
                schools[k][2]=round(student[1][k] 
                    + student[2][0], decimal_precision)
                break



#   Invoke second round DA-STB as required   #
def invoke_assignment_RLDA(students, schools, dropout_lotteries, numeric_delimiter, alpha):

    #   Calculate decimal precision    #
    decimal_precision = DECIMAL_PRECISION
    
    #   Capture relocation details  #
    relocated_student_lotteries, freshly_relocated = [], []

    #   DA-STB until seats fill     #
    cutoff_adjusted=0
    rerun_assignment=1
    
    while (rerun_assignment==1):
        
        #   Run DA_STB    #       
        freshly_relocated[:] = DA_STB_RLDA(students, schools, dropout_lotteries, numeric_delimiter, alpha)

        #   Check for empty seats in schools    #
        for i, school in enumerate(schools):
        
            if (school.count(numeric_delimiter) > 0):
            
                #   Adjust cutoff for the school    #
                adjust_cutoff_RLDA(students, schools, dropout_lotteries, i, decimal_precision, numeric_delimiter, alpha)
                cutoff_adjusted=1
        
        #   Check rerun condition   #
        if (cutoff_adjusted > 0):
            
            rerun_assignment=1
            cutoff_adjusted=0
            
        else:
        
            rerun_assignment=0
            
        relocated_student_lotteries.extend(freshly_relocated)
    
    return relocated_student_lotteries



#   Second round DA-STB  #
def DA_STB_RLDA(students, schools, dropout_lotteries, numeric_delimiter, alpha):

    #   Calculate decimal precision    #
    decimal_precision = DECIMAL_PRECISION
    
    #   Capture relocation details #
    relocated_student_lotteries, vacant_schools, interested_students = [], [], []
    
    #   Allocate per school    #
    for i in range(len(schools)):
    
        #   Find schools with newly generated vacancies     #
        vacant_schools[:] = [i for i in range(len(schools)) 
            if numeric_delimiter in schools[i]]
    
        for k in vacant_schools:
            
            #   Enlist eligible students     #
            interested_students[:] = [interested_student 
                for interested_student in students 
                if ((interested_student[2][0] not in dropout_lotteries)
                and i < len(interested_student[0]) - 1)]
            
            interested_students[:] = [interested_student 
                for interested_student in interested_students 
                if (interested_student[0][i] == k + 1)]
        
            for j, student in enumerate(sorted(interested_students, 
                key=lambda z: (-(z[:][1][k] + (z[:][3][0] + alpha * z[:][2][0]))))):
            
                #   Get private school position    #
                private_school_position=student[0].index(len(schools) + 1)
                
                #   Give proposals    #
                if (i < private_school_position):
                    
                    if (round(student[1][k] + (student[:][3][0] + alpha * student[:][2][0]), 
                        decimal_precision) >= schools[k][2] 
                        and numeric_delimiter in schools[k]):
                    
                        school_number, position_number = get_index(schools, student[2][0])
                        
                        if (school_number > -1 
                            and student[0].index(k + 1) < student[0].index(school_number + 1)):
                            
                            schools[school_number][position_number]=numeric_delimiter
                            
                            schools[k][schools[k].index(numeric_delimiter)]=student[2][0]
                            
                            relocated_student_lotteries.append(student[2][0])

                        else:
                            
                            schools[k][schools[k].index(numeric_delimiter)]=student[2][0]

    return relocated_student_lotteries



#   Adjust second round cutoff for the school    #
def adjust_cutoff_RLDA(students, schools, dropout_lotteries, index, decimal_precision, numeric_delimiter, alpha):

    #   Initialize current cutoff index     #
    current_cutoff_index=numeric_delimiter
    free_students = []
    
    #   Enlist unallocated students     #
    free_students[:] = [free_student for free_student in students 
        if (free_student[2][0] not in dropout_lotteries
        and get_index(schools, free_student[2][0])[0] < 0 
        and (index + 1) in free_student[0])]
    
    #   Loop by students to get required cutoff     #
    for i, student in enumerate(sorted(free_students, 
        key=lambda z: (-(z[:][1][index] + (z[:][3][0] + alpha * z[:][2][0]))))):
    
        #   Get current cutoff position     #
        if (round(student[1][index] + student[2][0], decimal_precision) 
            < schools[index][2] and current_cutoff_index < 0):
        
            current_cutoff_index=i
        
        #   Break loop if no reduction in cutoff can fill school seats     #
        if (i == len(free_students) - 1 
            and round(student[1][index] + (student[:][3][0] + alpha * student[:][2][0]), 
            decimal_precision) == schools[index][2]):
        
            schools[index][2]=round(student[1][index] 
                + student[2][0], decimal_precision)
            break
        
        #   Reduce cutoff to fill in school seats   #
        if (i == current_cutoff_index + schools[index].count(numeric_delimiter) - 1):
        
            schools[index][2]=round(student[1][index] 
                + student[2][0], decimal_precision)
            break



#   RLDA execution course   #
def RLDA_execution(alpha, students, RLDA_schools, data_store_location, dropout_lotteries, numeric_delimiter):

    RLDA_alpha_start = datetime.datetime.now()
    
    relocated_student_lotteries = []

    #   Second round school cutoff calculation  #
    calculate_cutoffs_RLDA(students, RLDA_schools, dropout_lotteries, numeric_delimiter, alpha)
    
    #   Second round DA-STB  #
    relocated_student_lotteries[:] = \
        invoke_assignment_RLDA(students, RLDA_schools, dropout_lotteries, numeric_delimiter, alpha)
    
    #   Track execution time    #
    RLDA_exec_time = datetime.datetime.now() - RLDA_alpha_start
    
    print_locked("\nUsing α = ", alpha, ", PLDA execution time ", 
        RLDA_exec_time, " hours, reassigned ", 
        len(relocated_student_lotteries), " students.\n\nTheir lottery numbers are:\n\n", 
        relocated_student_lotteries, sep="")
    
    #   Save data   #
    write_data_to_file(str(alpha) + "_RLDA_School", RLDA_schools, "Round II Allocations/")
    
    #   Return tracking data     #
    return [alpha, len(relocated_student_lotteries), RLDA_exec_time]



#   Interactive communication   #
def interaction(data_auto_generate, check_round, information, alpha, students, 
    schools, num_executions, numeric_delimiter, infinity):

    print_locked('\n\n\n\n{:.{align}{width}}'.format("Further Options", align='^', width=60))

    print_locked("\nPlease choose one of the following options to proceed:")
    
    print_locked('\n{:.{align}{width}}'.format("Options Start", align='^', width=60), end="\n\n")
    
    options = [
               "Gracefully Exit This Program",
               "View Last Run Statistics",
               "Rerun This Program Afresh",
               "Rerun Round II Of This Program",
               "Rerun Round II With Different Alpha(α) Values",
               "View Students Data",
               "View Round I School Allocations",
               "View Round II FLDA School Allocations",
               "View Round II PLDA School Allocations",
               "View Execution Results"
               ]
    
    [print_locked("\n\t", count, " --> ", item, sep="") for count, item in enumerate(options)]
    
    print_locked('\n{:.{align}{width}}'.format("End of Options", align='^', width=60))
    
    print_locked("\nPlease enter your option from the above: ", end="")
    response = input()
    
    try:
        response = int(response)

        if  (response == 0):
            print_locked("\n\nQuitting program..\n\n")
            sys.exit()
            
        elif (response == 1):
            print_locked("\n\nDisplaying statistics again..\n\n")
            print_timer_data = multiprocessing.Process(target=print_information, 
            args=(data_auto_generate, check_round, information, len(students), 
                  len(schools), num_executions, num_executions, infinity))
            print_timer_data.start()
            print_timer_data.join()
            print_locked("\n\nChoice action completed.\n\n")
            
        elif (response == 2):
            print_locked("\n\nRerunning program afresh..\n\n")
            main(1)

        elif (response == 3):
            print_locked("\n\nRerunning Round II of the program..\n\n")
            main(2, information)
            print_locked("\n\nChoice action completed.\n\n")


        elif (response == 4):
            print_locked("\n\nPlease enter a new Alpha(α) range limits'"\
            + " space-separated values (α ∈ [min_value, max_value]/[-value, value]): ", end="")
            
            try:
            
                limit_alpha = input()
                
                if (limit_alpha == ""):
                
                    raise Exception
                    
                range_alpha = [int(value) for value in limit_alpha.strip().split()]
            
            except Exception:
            
                range_alpha = [min(alpha), max(alpha)]
                print_locked("\nAlpha(α) values defaulted to α ∈ [", 
                min(range_alpha), ", ", max(range_alpha), "].", sep="")
                
            if (len(range_alpha) == 1):
            
                range_alpha = [-range_alpha[0], range_alpha[0]]
            
            print_locked("\n\nRerunning Round II of the program with updated Alpha(α) values..\n\n")
            main(2, information, alpha = list(range(min(range_alpha), max(range_alpha) + 1)))
            print_locked("\n\nChoice action completed.\n\n")

        elif (response == 5):
            print_locked("\n\nPrinting student data..\n\n")
            print_student_data = multiprocessing.Process(target=custom_print, 
                args=("Students", students))
            print_student_data.start()
            print_student_data.join()
            print_locked("\n\nChoice action completed.\n\n")

        elif (response == 6):
            print_locked("\n\nPrinting Round I school data..\n\n")
            round_1_schools = load_school_preferences(schools)
            print_school_data = multiprocessing.Process(target=custom_print, 
                args=("Schools", round_1_schools))
            print_school_data.start()
            print_school_data.join()
            print_locked("\n\nChoice action completed.\n\n")

        elif (response == 7):
            print_locked("\n\nPrinting Round II PLDA school data..\n\n")
            print_locked("\n\nPlease customize this option.\n\n")
            print_locked("\n\nChoice action completed.\n\n")

        elif (response == 8):     
            print_locked("\nPlease enter the corresponding Alpha(α) value of the allocation (α ∈ ["
                + str(START_ALPHA) + ", " + str(END_ALPHA) + "]): ", sep="", end="")
            display_alpha = int(input())
            
            round_2_schools = load_school_preferences(schools, 
                str(display_alpha) + "_RLDA", "Round II Allocations/")
            
            print_locked("\n\nPrinting Round II PLDA school data with Alpha(α) value",
                display_alpha, "..\n\n")
            
            print_school_data = multiprocessing.Process(target=custom_print, 
                args=("Schools", round_2_schools))
            print_school_data.start()
            print_school_data.join()
            
            print_locked("\n\nChoice action completed.\n\n")

        elif (response == 9):
            print_locked("\n\nDisplaying execution results..\n\n")
            print_execution_results = \
                multiprocessing.Process(target=display_execution_results, 
                args=(DATA_STORE_LOCATION, num_executions, numeric_delimiter))
            print_execution_results.start()
            print_execution_results.join()
            print_locked("\n\nChoice action completed.\n\n")

        else:
            print_locked("\n\nPlease enter a valid option..\n\n")

    except Exception:
        print_locked("\n\nPlease enter a valid option..\n\n")
        print_locked(traceback.format_exc())



#   Round I execution   #
def round_1(cpu_count, preset_factor, numeric_delimiter, data_store_location):

    #   Reset tracker   #
    students = []
    schools = []
    information = []
    
    #   Auto-generate preference data   #
    data_auto_generate = get_data_generation_options()
    
    num_executions = get_num_executions()
    
    if data_auto_generate.upper() == 'Y':
    
        #   Take numbers of students and schools as inputs  #
        student_count, school_count = input_numbers()
        
        #   Reset data generation option if input data is not valid   #
        if (student_count < 0):
        
            data_auto_generate = ""
    
    #   Generate data   #
    if data_auto_generate.upper() == 'Y':
  
        #   Track whole code performance    #
        initiate = datetime.datetime.now()
        information.append(initiate.time())
        
        #   Set round value #
        check_round = 1
        
        #   Generate school preferences    #
        start = datetime.datetime.now()
        
        schools = pool.map(partial(generate_per_school, 
        student_count = student_count, school_count = school_count), 
        list(range(school_count)),
        chunksize = math.ceil(school_count/cpu_count))

        school_generation_time = datetime.datetime.now() - start
        information.append(school_generation_time)
        
        print_locked("\nSchool generation time =", 
        school_generation_time, "hours")
        
        #   Save school data     #
        pool.apply_async(write_data_to_file, args=("School", schools))
        
        #   Generate student data   #
        student_start = datetime.datetime.now()
        
        #   Create a temporary folder..Delete and recreate if exists  #   
        try:
            shutil.rmtree(data_store_location + ".TempData/")
        except Exception:
            pass
        os.makedirs(data_store_location + ".TempData/")
        
        queue = multiprocessing.Queue()
        
        generate_student = multiprocessing.Process(target=generate_student_data, 
        args=(student_count, school_count, cpu_count, preset_factor, schools, queue))
        
        generate_student.start()
        information.append(queue.get())
        lottery = queue.get()
        generate_student.join()
        
        #   Get total student data from temporary files   #
        if (student_count < preset_factor ** 2):
        
            print_locked("\nLoading saved student data..")
            students = get_partial_student_data(0, 1)
            
        else:
        
            print_locked("\nLoading saved student data in", cpu_count, "parts..")
            
            students = \
            [item for sublist in pool.map(partial(get_partial_student_data, 
            total_parts = cpu_count), range(cpu_count)) for item in sublist]
            
        print_locked("\nData load complete..")
        
        shutil.rmtree(data_store_location + ".TempData/")
        
        student_generation_time = datetime.datetime.now() - student_start
        information.append(student_generation_time)
        
        print_locked("\nStudent data generation time =", 
        student_generation_time, "hours")
        
        # Save student data #
        multiprocessing.Process(target=write_data_to_file, 
        args=("Student", students)).start()
    
    else:
        
        #   Option for starting from the second round directly  #
        check_round = check_round_to_start_with()
        
         #   Track whole code performance    #
        initiate = datetime.datetime.now()
        information.append(initiate.time())
        
        #   Load existing data  #
        start = datetime.datetime.now()
        
        lottery, students = load_student_preferences(students)
        schools = load_school_preferences(schools)
        
        student_count = len(students)
        school_count = len(schools)
        
        data_load_time = datetime.datetime.now() - start
        information.extend([data_load_time, None, None])
        
        print_locked("\nData load time =", 
        data_load_time, "hours")
        
        if (check_round == 1):
        
            schools[:] = [school[:2] for school in schools]
    
    if (check_round == 1):
    
        print_locked("\n\nStarting Round I execution..")
    
        #   First round school cutoff calculation  #
        start = datetime.datetime.now()
        
        calculate_cutoffs(students, schools)
        
        cutoff_generation_time = datetime.datetime.now() - start
        information.append(cutoff_generation_time)
        
        print_locked("\nCutoff generation time =", 
        cutoff_generation_time, "hours")
        
        #   First round DA-STB  #
        time_taken, schools = invoke_assignment(students, schools, cpu_count, numeric_delimiter)
        
        information.append(time_taken)
        
        #   Save school data after first round   #
        multiprocessing.Process(target=write_data_to_file, 
        args=("School", schools)).start()

    else:
    
        information.extend([None, None])

    return students, schools, student_count, school_count, check_round, \
           information, initiate, data_auto_generate, num_executions, lottery



#   Round II customizations  #
def round_2_customizations(information):

    students = []
    schools = []

    #   Update trackers     #
    initiate = datetime.datetime.now()
    data_auto_generate = ""
    check_round = 2
    del information[6:]
    information[0] = str(initiate.time())
        
    #   Load existing data  #
    start = datetime.datetime.now()
    
    lottery, students = load_student_preferences(students)
    schools = load_school_preferences(schools)
    
    student_count = len(students)
    school_count = len(schools)
    
    data_load_time = datetime.datetime.now() - start
    print_locked("\nData load time =", 
    data_load_time, "hours")
    information[1:4] = [data_load_time, None, None]
    
    #   Get number of executions    #
    num_executions = get_num_executions()

    return data_auto_generate, information, students, schools, lottery, \
           check_round, initiate, student_count, school_count, num_executions


#   Round II execution   #
def round_2(students, schools, student_count, school_count, cpu_count, 
    current_execution, check_round, numeric_delimiter, information, initiate, 
    data_auto_generate, num_executions, alpha, data_store_location, lottery, infinity):

    print_locked("\n\nStarting Round II execution..")
    
    dropout_lotteries, schools_round_2, FLDA_schools, RLDA_schools = [], [], [], []

    #   Remove 10% students as is the trend for dropouts    #
    dropout_lotteries[:] = [item for sublist in 
                         random.sample(lottery, int(0.1 * student_count))
                         for item in sublist]
    
    schools_round_2[:] = pool.map(partial(dropouts,
        dropout_lotteries = dropout_lotteries, 
        numeric_delimiter = numeric_delimiter), 
        schools, chunksize = math.ceil(school_count/cpu_count))

    print_locked("\nTo simulate approx. 10% dropouts, ", len(dropout_lotteries), 
    " random students were dropped.\n\nTheir lottery numbers are\n\n", dropout_lotteries, sep="")
    
    #   Updating priorities of allocated students   #
    students[:] = \
        pool.map(partial(update_priorities, schools = schools_round_2), students)
    
    print_locked("\n\nPriorities of allocated students updated for Round II allocations.")
    
    #   RLDA execution  #
    RLDA_start = datetime.datetime.now()
    
    #   Create Round II Allocations folder..Delete and recreate if exists  #   
    try:
        shutil.rmtree(data_store_location + "Round II Allocations/")
    except Exception:
        pass
    
    os.makedirs(data_store_location + "Round II Allocations/")
    
    print_locked("\nStarting Round II PLDA executions using all available values in Alpha(α) ∈ Integers(Z) and Alpha(α) ∈ [",
        min(alpha), ", ", max(alpha), "] and Alpha(α) = (-∞, ∞)", sep="")
    
    RLDA_alphas_relocations_times = []
    
    RLDA_alphas_relocations_times[:] = \
        pool.imap_unordered(partial(RLDA_execution, students = students, RLDA_schools = copy.deepcopy(schools_round_2), 
        dropout_lotteries = dropout_lotteries, numeric_delimiter = numeric_delimiter,
        data_store_location = data_store_location), [infinity, -infinity] + alpha, 
        chunksize = 1)
    
    RLDA_alphas_relocations_times.sort(key=lambda x: -x[0])
    
    #   Add information from Round II assignments  #
    information.extend([RLDA_alphas_relocations_times])
    
    RLDA_exec_time = datetime.datetime.now() - RLDA_start
    print_locked("\nTotal PLDA execution time:", RLDA_exec_time, "hours")
    information.extend([RLDA_exec_time])

    #   Save execution results     #
    multiprocessing.Process(target=write_execution_results, 
        args=(current_execution, student_count, school_count, 
        numeric_delimiter, information[-2])).start()
    
    #   Completion timer    #
    program_completion_time = datetime.datetime.now() - initiate
    information.append(program_completion_time)
    
    #   Add dropout data    #
    information.append(dropout_lotteries)
    
    #   Print tracked time data     #
    print_statistics = \
        multiprocessing.Process(target=print_information, 
        args=(data_auto_generate, check_round, information, student_count, 
        school_count, num_executions, current_execution, infinity))
    
    print_statistics.start()
    print_statistics.join()

    return information



#   Notify program completion   #
def completion_alert():

    duration = 1    # Seconds
    freq = 1000     # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('spd-say "Execution complete."')



##  The main function   ##

#   Main    #
def main(excute_round_number = 1, information = [], students = [], schools = [], alpha = []):

    global START_ALPHA, END_ALPHA

    #   Setup preferences   #
    if (len(alpha) == 0):
    
        alpha               = list(range(-ALPHA, ALPHA + 1))
        START_ALPHA         = -ALPHA
        END_ALPHA           = ALPHA
        
    else:
    
        START_ALPHA         = min(alpha)
        END_ALPHA           = max(alpha)

    factor              = FACTOR
    infinity            = INFINITY
    cpu_count           = CPU_COUNT
    numeric_delimiter   = NUMERIC_DELIMITER
    data_store_location = DATA_STORE_LOCATION
    
    #   Round I start  #
    if (excute_round_number == 1):
    
        students, schools, student_count, school_count, check_round, \
        information, initiate, data_auto_generate, num_executions, lottery = \
        round_1(cpu_count, factor, numeric_delimiter, data_store_location)
    
    #   Round II customizations  #
    if (excute_round_number == 2):
    
        data_auto_generate, information, students, schools, lottery, \
        check_round, initiate, student_count, school_count, num_executions = \
        round_2_customizations(information)
    
    #   Round II start  #
    for i in range(num_executions):
    
        #   Reset trackers for multiple Round II executions  #
        if (i > 0):
        
            del information[6:]
    
        #   Execute Round II allocations     #
        information = \
        round_2(students, schools, student_count, school_count, cpu_count, i + 1, 
        check_round, numeric_delimiter, information, initiate, data_auto_generate, 
        num_executions, alpha, data_store_location, lottery, infinity)
    
    #   Announce program finish #
    pool.apply_async(completion_alert, args=())
    
    #   Save execution results as a graph pdf   #
    multiprocessing.Process(target=display_graph, args=(data_store_location, )).start()
    
    #   Interactive options     #
    
    if (excute_round_number == 1):
    
        while (1):
        
            interaction(data_auto_generate, check_round, information, alpha,
                        students, schools, num_executions, numeric_delimiter, infinity)



##  Call the main function  ##

#   Initiation  #
if __name__=="__main__":

    try:
    
        #   Start logging to file     #        
        print_locked('\n\n\n\n{:.{align}{width}}'.format("Fresh Execution Start at: " 
            + str(datetime.datetime.now()), align='^', width=100), end="\n\n")
        
        print_locked("\n\nProgram Name:\n\n" + str(sys.argv[0].split("/")[-1]))
        
        print_locked("\n\nProgram Path:\n\n" + os.path.dirname(sys.argv[0]))
        
        print_locked("\n\nProgram Name With Path:\n\n" + str(sys.argv[0]), end="\n\n\n")
        
        #   Clear the terminal  #
        os.system("clear")
        
        #   Initiate lock object    #
        lock = multiprocessing.Lock()

        #   Initiate pool objects   #
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        
        #   Call the main program   #
        main()
        
        #   Close Pool object    #
        pool.close()

    except Exception:
    
        print_locked(traceback.format_exc())


##   End of Code   ##

