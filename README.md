# CSCI 8360 Data Science Practicum : Project NeuroFinder

  ## Goal
  To develop a segmentation pipeline that identifies the regions of as many neurons in the video as possible, as accurately as possible.
  
  ## Getting Started
  These instructions describe the prerequisites and steps to get the project up and running.

  ### Setup
  This project can be easily set up on the Google Cloud Platform, using a '[Compute Engine VM Instance](https://console.cloud.google.com/compute/instances)' or a '[Deep Learning VM Instance](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning)' with/without a GPU. You will need to have [Google Cloud SDK](https://cloud.google.com/sdk/install) installed in your local machine to be able to set the project up.

  After downloading/cloning this repository to your local machine, the user will need to open the `Google Cloud SDK Shell`. Once it opens, the user can copy the contents of this repository to the Deep Learning VM instance using the command:

  `gcloud compute scp --recurse /complete/link/to/repository/* <user>@<instance_name-vm>:/home/<user>/`

  Furthermore, to setup a VM instance with all the prerequisite packages used in the project, do the following:
  1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
  2. Create a conda environment from the included `environment.yml` file using the following command:
     
     `$ conda env create -f environment.yml`
  3. Activate the environment
     
     `$ source activate hopper`

  ### Usage
  To run the code and generate output prediction masks in the `/results` directory, the user can navigate to the folder containing the file 'team-hopper.py', and run it using the command: `$ python team-hopper.py --options`. The user can get a description of the options by using the command: `$ python team-hopper.py -help`.
  
  ### Dataset
  The dataset was created by [codeneuro.org](http://codeneuro.org/). 
  ![alt text](http://url/to/img.png) ![alt text](http://url/to/img.png)
  The image on the left is a training example and on the right is the mask with the regions circled.
  
  ### Approach
  NMF
  UNET
  NMF plus UNET
  PCA
  
  ### Output
  Upon running the command in the ‘Usage’ section, the dataset will be downloaded from the Google Storage bucket link carrying the Neuron dataset, and the output `submission.json` files will be generated in the `/results` directory. Here's a preview of the `submission.json` file:
 ```
  [                                                   
      {  "dataset": "00.01.test",                      
         "regions":
            [
              {"coordinates": [ [0,0], [0,1] ]},
              {"coordinates": [ [10,12], [14,17] ]},
            ]                                           
      }                                                    
  ]
  ```

  ## Contributors
* See [Contributors](CONTRIBUTORS.md) file for more details.

## License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for more details.
