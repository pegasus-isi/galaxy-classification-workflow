#!/usr/bin/env python3

import os
from Pegasus.api import *
from pathlib import Path
import logging
import time
import argparse
from bin.GalaxyDataset import GalaxyDataset

logging.basicConfig(level=logging.DEBUG)

#full dataset
MAX_IMG_0 = 8436
MAX_IMG_1 = 8069
MAX_IMG_2 = 579
MAX_IMG_3 = 3903
MAX_IMG_4 = 7806


def split_preprocess_jobs(preprocess_images_job, input_images, postfix):
    
    resized_images = []
    
    for i in range(len(input_images)):
        curr = i % len(preprocess_images_job)
        preprocess_images_job[curr].add_inputs(input_images[i])
        out_file = File(str(input_images[i]).split(".")[0] + postfix + ".jpg")
        preprocess_images_job[curr].add_outputs(out_file)
        resized_images.append(out_file)
        
    return resized_images


def add_augmented_images(class_str, num, start_num):
    augmented_files = []
    for i in range(num):
        augmented_files.append(File("train_" + class_str + "_" + str(start_num) + "_proc.jpg"))
        start_num +=1
    return augmented_files


def create_files_hpo(input_files):
    files = []
    for file in input_files:
        name = File(file.split("/")[-1].split(".")[0] + "_proc.jpg")
        files.append(name)
    return files


def run_workflow(DATA_PATH):
    props = Properties()
    props["pegasus.mode"] = "development"
    props["pegasus.transfer.links"] = "true"
    props["pegasus.transfer.threads"] = "8"
    props["pegasus.monitord.encoding"] = "json"
    props["pegasus.catalog.workflow.amqp.events"] = "stampede.*"
    props["pegasus.catalog.workflow.amqp.url"] = "amqp://panorama:panorama@iris.isi.edu:5674/panorama/monitoring"
    props["pegasus.transfer.stageout.remote.sites"] = "donut"
    
    props.write()

    ### ADD SITE CATALOG
    #-------------------------------------------------------------------------------------------------------
    sc = SiteCatalog()

    shared_scratch_dir = os.path.join(os.getcwd(), "scratch")
    local_storage_dir = os.path.join(os.getcwd(), "output")

    local = Site("local")\
                .add_directories(
                    Directory(Directory.SHARED_SCRATCH, shared_scratch_dir)
                        .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                    Directory(Directory.LOCAL_STORAGE, local_storage_dir)
                        .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                )

    donut = Site("donut")\
                .add_grids(
                    Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM, contact="${DONUT_USER}@donut-submit01", job_type=SupportedJobs.COMPUTE),
                    Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM, contact="${DONUT_USER}@donut-submit01", job_type=SupportedJobs.AUXILLARY)
                )\
                .add_directories(
                    Directory(Directory.SHARED_SCRATCH, "/nas/home/${DONUT_USER}/pegasus/scratch")
                        .add_file_servers(FileServer("file://${DONUT_USER}@donut-submit01/nas/home/${DONUT_USER}/pegasus/scratch", Operation.GET))\
                        .add_file_servers(FileServer("scp://${DONUT_USER}@donut-submit01/nas/home/${DONUT_USER}/pegasus/scratch", Operation.PUT)),
                    Directory(Directory.SHARED_STORAGE, "/nas/home/${DONUT_USER}/pegasus/storage")
                        .add_file_servers(FileServer("file://${DONUT_USER}@donut-submit01/nas/home/${DONUT_USER}/pegasus/storage", Operation.ALL))
                )\
                .add_pegasus_profile(
                    style="ssh",
                    data_configuration="nonsharedfs",
                    change_dir="true",
                    queue="donut-default",
                    cores=8,
                    runtime=4800,
                    grid_start_arguments="-m 10"
                )\
                .add_profiles(Namespace.PEGASUS, key="SSH_PRIVATE_KEY", value="${HOME}/.ssh/bosco_key.rsa")\
                .add_env(key="PEGASUS_HOME", value="${DONUT_USER_HOME}/${PEGASUS_VERSION}")\
                .add_env(key="PEGASUS_TRANSFER_PUBLISH", value="1")\
                .add_env(key="PEGASUS_AMQP_URL", value="amqp://panorama:panorama@iris.isi.edu:5674/panorama/monitoring")\
                .add_env(key="KICKSTART_MON_URL", value="rabbitmq://panorama:panorama@iris.isi.edu:15674/api/exchanges/panorama/monitoring/publish")
        
    sc.add_sites(local, donut)
    sc.write()

    ### ADD INPUT FILES TO REPILCA CATALOG
    #-------------------------------------------------------------------------------------------------------
    rc = ReplicaCatalog()

    metadata_file      = 'config/training_solutions_rev1.csv'
    available_images     = 'config/full_galaxy_data.log'
    
    galaxy_dataset = GalaxyDataset([MAX_IMG_0, MAX_IMG_1, MAX_IMG_2, MAX_IMG_3, MAX_IMG_4], SEED, metadata_file, available_images)
    dataset_mappings = galaxy_dataset.generate_dataset()
    print(len(dataset_mappings))
    
    output_images = []
    output_files = []
    for m in dataset_mappings:
        output_images.append(m[1])
        output_files.append(File(m[1]))
        rc.add_replica("donut", m[1], os.path.join(DATA_PATH, m[0]))

    # ADDITIONAL PYTHON SCRIPS NEEDED BY TUNE_MODEL
    #-------------------------------------------------------------------------------------------------------
    data_loader_fn = "data_loader.py"
    data_loader_file = File(data_loader_fn )
    rc.add_replica("local", data_loader_fn, os.path.join(os.getcwd(), "bin/" + data_loader_fn ))

    model_selction_fn = "model_selection.py"
    model_selction_file = File(model_selction_fn )
    rc.add_replica("local", model_selction_fn, os.path.join(os.getcwd(),"bin/" + model_selction_fn ))


    # FILES FOR vgg16_hpo.py VGG 16
    #--------------------------------------------------------------------------------------------------------
    vgg16_pkl_file = File("hpo_galaxy_vgg16.pkl")
    rc.add_replica("local", vgg16_pkl_file, os.path.join(os.getcwd(), "config", vgg16_pkl_file.lfn))    

    # FILES FOR train_model.py 
    #--------------------------------------------------------------------------------------------------------
    checkpoint_vgg16_pkl_file = File("checkpoint_vgg16.pkl")
    rc.add_replica("local", checkpoint_vgg16_pkl_file, os.path.join(os.getcwd(), "config", checkpoint_vgg16_pkl_file.lfn))

    rc.write()

    # TRANSFORMATION CATALOG
    #---------------------------------------------------------------------------------------------------------
    tc = TransformationCatalog()


    # define container for the jobs
    galaxy_container = Container(
                'galaxy_container',
                Container.SINGULARITY,
                image = str(Path(".").parent.resolve() / "containers/galaxy_container.sif"),
                image_site = "local",
                mounts=["${DONUT_USER_HOME}:${DONUT_USER_HOME}"]
    ).add_env(TORCH_HOME="/tmp")

    # Data preprocessing part 1: image resize
    preprocess_images = Transformation("preprocess_images", site="local",
                                    pfn = str(Path(".").parent.resolve() / "bin/preprocess_resize.py"), 
                                    is_stageable= True, container=galaxy_container)

    # Data preprocessing part 2: image augmentation
    augment_images = Transformation("augment_images", site="local",
                                    pfn = str(Path(".").parent.resolve() / "bin/preprocess_augment.py"), 
                                    is_stageable= True, container=galaxy_container)

    # HPO: main script
    vgg16_hpo = Transformation("vgg16_hpo",
                   site="local",
                   pfn = str(Path(".").parent.resolve() / "bin/vgg16_hpo.py"), 
                   is_stageable= True,
                   container=galaxy_container
                )\
                .add_pegasus_profile(cores=16, gpus=1, runtime=14400, grid_start_arguments="-G -m 10")\
                .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    # Train Model
    train_model = Transformation("train_model",
                      site="local",
                      pfn = str(Path(".").parent.resolve() / "bin/train_model_vgg16.py"), 
                      is_stageable= True, 
                      container=galaxy_container
                  )\
                  .add_pegasus_profile(cores=16, gpus=1, runtime=14400, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_UTIL", value="TRUE")

    # Eval Model
    eval_model = Transformation("eval_model",
                     site="local",
                     pfn = str(Path(".").parent.resolve() / "bin/eval_model_vgg16.py"), 
                     is_stageable= True, 
                     container=galaxy_container
                 )\
                .add_pegasus_profile(cores=8, gpus=1, runtime=600, grid_start_arguments="-G -m 10")\
                .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    tc.add_containers(galaxy_container)
    tc.add_transformations(
        preprocess_images,
        augment_images,
        vgg16_hpo,
        train_model,
        eval_model
        )
    tc.write()

    ## CREATE WORKFLOW
    #---------------------------------------------------------------------------------------------------------
    wf = Workflow('Galaxy-Classification-Workflow-Donut')

    job_preprocess_images = [Job(preprocess_images) for i in range(NUM_WORKERS)]
    resized_images = split_preprocess_jobs(job_preprocess_images, output_files, "_proc")

    train_class_2         = "train_class_2"
    train_files_class_2   = [i for i in output_images if train_class_2 in i]
    input_aug_class_2     = [ File(file.split("/")[-1].split(".")[0] + "_proc.jpg") for file in train_files_class_2 ]
    output_aug_class_2    = add_augmented_images("class_2", NUM_CLASS_2, 4000)
    
    train_class_3         = "train_class_3"
    train_files_class_3   = [i for i in output_images if train_class_3 in i]
    input_aug_class_3     = [ File(file.split("/")[-1].split(".")[0] + "_proc.jpg") for file in train_files_class_3 ]
    output_aug_class_3    = add_augmented_images("class_3", NUM_CLASS_3, 4000)
    

    job_augment_class_2 = Job(augment_images)\
                        .add_args("--class_str class_2 --num {}".format(NUM_CLASS_2))\
                        .add_inputs(*input_aug_class_2)\
                        .add_outputs(*output_aug_class_2)

    job_augment_class_3 = Job(augment_images)\
                        .add_args("--class_str class_3 --num {}".format(NUM_CLASS_3))\
                        .add_inputs(*input_aug_class_3)\
                        .add_outputs(*output_aug_class_3)


    train_class = 'train_class_'
    train_class_files = [i for i in output_images if train_class in i]
    val_class = 'val_class_'
    val_class_files = [i for i in output_images if val_class in i]
    test_class = 'test_class_'
    test_class_files = [i for i in output_images if test_class in i]


    input_hpo_train = create_files_hpo(train_class_files)
    input_hpo_val   = create_files_hpo(val_class_files)
    input_hpo_test  = create_files_hpo(test_class_files)


    best_params_file = File("best_vgg16_hpo_params.txt")

    # Job HPO
    job_vgg16_hpo = Job(vgg16_hpo)\
                        .add_args("--trials {} --epochs {} --batch_size {}".format(TRIALS, EPOCHS, BATCH_SIZE))\
                        .add_inputs(*output_aug_class_3, *output_aug_class_2,\
                            *input_hpo_train, *input_hpo_val,data_loader_file, model_selction_file)\
                        .add_checkpoint(vgg16_pkl_file, stage_out=True)\
                        .add_outputs(best_params_file)


    # Job train model
    job_train_model = Job(train_model)\
                        .add_args("--epochs {} --batch_size {}".format( EPOCHS, BATCH_SIZE))\
                        .add_inputs(*output_aug_class_3, *output_aug_class_2, best_params_file,\
                            *input_hpo_train, *input_hpo_val, *input_hpo_test,\
                            data_loader_file, model_selction_file)\
                        .add_checkpoint(checkpoint_vgg16_pkl_file , stage_out=True)\
                        .add_outputs(File("final_vgg16_model.pth"),File("loss_vgg16.png"))


    # Job eval
    job_eval_model = Job(eval_model)\
                        .add_inputs(*input_hpo_test,data_loader_file,best_params_file,\
                                    model_selction_file,File("final_vgg16_model.pth"))\
                        .add_outputs(File("final_confusion_matrix_norm.png"),File("exp_results.csv"))


    ## ADD JOBS TO THE WORKFLOW
    wf.add_jobs(*job_preprocess_images,job_augment_class_2 ,job_augment_class_3, job_vgg16_hpo,\
                job_train_model,job_eval_model)  


    # EXECUTE THE WORKFLOW
    #-------------------------------------------------------------------------------------
    try:
        wf.plan(submit=False, sites=["donut"], output_sites=["donut"])
        #wf.wait()
        #wf.statistics()
    except PegasusClientError as e:
        print(e.output)   
    #graph_filename = "galaxy-wf.dot"
    #wf.graph(include_files=True, no_simplify=True, label="xform-id", output = graph_filename)


def main():
    
    start = time.time()
    
    global ARGS
    global BATCH_SIZE
    global SEED
    global DATA_PATH
    global EPOCHS
    global TRIALS
    global NUM_WORKERS
    global NUM_CLASS_2
    global NUM_CLASS_3

    
    parser = argparse.ArgumentParser(description="Galaxy Classification")   
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--seed', type=int, default=10, help='select seed number for reproducibility')
    parser.add_argument('--data_path', type=str, default='full_galaxy_data/',help='path to dataset ')
    parser.add_argument('--epochs', type=int,default=10, help = "number of training epochs")  
    parser.add_argument('--trials', type=int,default=1, help = "number of trials") 
    parser.add_argument('--num_workers', type=int, default= 20, help = "number of workers")
    parser.add_argument('--num_class_2', type=int, default= 7000, help = "number of augmented class 2 files")
    parser.add_argument('--num_class_3', type=int, default= 4000, help = "number of augmented class 3 files")

    


    ARGS        = parser.parse_args()
    BATCH_SIZE  = ARGS.batch_size
    SEED        = ARGS.seed
    DATA_PATH   = ARGS.data_path
    EPOCHS      = ARGS.epochs
    TRIALS      = ARGS.trials
    NUM_WORKERS = ARGS.num_workers
    NUM_CLASS_2 = ARGS.num_class_2
    NUM_CLASS_3 = ARGS.num_class_3

    run_workflow(DATA_PATH)
    
    exec_time = time.time() - start

    print('Execution time in seconds: ' + str(exec_time))
    

    return

if __name__ == "__main__":
    
    main()
