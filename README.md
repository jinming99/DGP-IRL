# DGP-IRL
Deep Gaussian Process for Inverse Reinforcement Learning

To run the algorithm, please first download the toolbox by Levine et. al. (2010):
http://graphics.stanford.edu/projects/gpirl/irl_toolkit.zip

Then put the folder of deepGPIRL and Binaryworld in the same folder, and add the paths to these folders. You don't need to specify the parameters to start with. You can modify them in the file deepgpirldefaultparams.m .

Example to run the DGP-IRL algorithm on binary world benchmark:
test_result_dgpirl = runtest('deepgpirl',struct(),'linearmdp','binaryworld',struct('n',12),struct('training_sample_lengths',12^2,'training_samples',8,'verbosity',1));
