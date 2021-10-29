import os


def main():
    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    for i in range(len(learning_rates)):
        ff_learning_rate_run(learning_rates[i], i)


def ff_learning_rate_run(learning_rate, num):
    command = "python3 main.py train --data-dir data --log-file logs/ff-logs" + str(num) + \
              ".csv --model-save modelfiles/ff" + str(num) + ".torch --model simple-ff --learning-rate" + \
              str(learning_rate)
    os.system(command)


def ff_batch_size_run(batch_size, num):
    command = "python3 "
    file = "main.py "
    mode = "train "
    data_dir = "--data-dir data "
    log_file = "--log-file logs/ff-logs" + str(num) + ".csv "
    model_save = "--model-save modelfiles/ff" + str(num) + ".torch "
    model = "--model simple-ff "
    batches = "--batch-size " + str(batch_size)
    full_command = command + file + mode + data_dir + log_file + model_save + model + batches
    os.system(full_command)


if __name__ == '__main__:':
    main()
