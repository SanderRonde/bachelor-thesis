# Script usage

All files contain a bunch of all caps constants at the top of the script. Change these in order to for example enable verbose mode, logging, skipping sections, changing the architecture, or others.

There are three stages, preprocessing, training/testing and translating. The following sections describe which files belong what stage. 

## Preprocessing

The file ```dataset_generator.py``` is used for preprocessing. Usage example:

```bash
python3 dataset_generator.py -i dataset_file.h5 -o output_features.p -c 10 -n dataset_name -p 100.0
```

Or simply run 
```bash
python3 dataset_generator.py -h
```
for an overview. 

Parallelizing can be done by increasing the amount specified by ```-c```. This increases the amount of CPUs assigned to the task.

## Training/testing

The files ```split_work.py``` and ```main.py``` are used in training and testing.


### split_work.py

 ```split_work.py``` serves as a method to parallelize ```main.py``` when it's using GPUs. ```split_work.py``` usage example:

```bash
python3 split_work.py -g 16 -l logs.log -c "[[main.py command]]"
```

where ```"[[main.py command]]"``` contains a valid ```main.py``` command. The above example will run on 16 GPUs (as ```-g``` indicates). ```split_work.py``` opens a new tmux window, which it splits into 16 (or however many GPUs you are using) panes, each running 1/16th of the total work (or a different number depending on how many GPUs you use). You can also run:

```bash
python3 split_work.py -h
```
for help regarding use.

### main.py

```main.py``` does the main job of training and testing. Keep in mind that when specifying a valid ```main.py``` command for the command above, the ```python3 main.py``` part needs to be included, not just its arguments.

Usage example:

```bash
python3 main.py -i features.p -o anomalies.encoded.json -p plot_dir/ -l logs_dir/ -e experiment_name
```

or just run

```bash
python3 main.py -h
```

for help.

## Translating

The file ```translate_anomalies.py``` is used to translate anomalies back into human-readable events. Simply input the encoded file ```main.py``` produced and the dataset file and specify an output location.

Usage example:

```bash
python3 translate_anomalies.py -i anomalies.encoded.json -o anomalies.json -d dataset_file.h5
```

or consult ```-h``` again.


# Full example

When running these commands, the anomalies can be found in ```anomalies.json```.


## GPU example:
```bash
python3 dataset_generator.py dataset.h5 -o features.p && \
python3 split_work.py -l logs/split_work_logs.log -g 16 -c "python3 main.py -i features.p -o anomalies.encoded.json -p plots/ -l logs/" && \
python3 translate_anomalies.py -i anomalies.encoded.json -o anomalies.json -d dataset.h5
```

## CPU example:
```bash
python3 dataset_generator.py dataset.h5 -o features.p && \
python3 main.py -i features.p -o anomalies.encoded.json -p plots/ -l logs/ && \
python3 translate_anomalies.py -i anomalies.encoded.json -o anomalies.json -d dataset.h5
```
Keep in mind that no plots are actually plotted in this example, only data files with which plots can be made are generated. To do that, run ```split_work.py``` afterwards with the ```SKIP_MAIN``` constant set to true like this:

```bash
python3 split_work.py -g 0 -c "python3 main.py -p plots/"
```