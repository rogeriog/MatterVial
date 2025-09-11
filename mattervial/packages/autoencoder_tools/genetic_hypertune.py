def genetic_hypertune_autoencoder():
   print("This is just a placeholder for public implementation later.")
# import pygad
# import numpy as np
# from autoencoder_tools.autoencoder_setup import train_autoencoder

# def genetic_hypertune_autoencoder(prefix_name = 'testmodel',
#     dataset=None,
#     compress_ratio = 0.2,
#     architecture={'arch':'custom_VAE'},
#     gene_space = [[0.5,1.0,1.5,2.0,2.5],list(range(10,200,20)),[0.001,0.0005],[8,16,32,64],[0,1,2]],
#     ga_settings = {'num_generations': 100, 'num_parents_mating': 4, 
#     'sol_per_pop': 10, 'parent_selection_type': 'rws', 'keep_elitism': 1},
#     initial_population = None,
#     savedir = './tmp/',
#     logfile = 'EncoderResults.txt',
#     random_state = 1,
#     ):
#     '''
#     This function uses pygad to optimize the hyperparameters of an autoencoder with a genetic algorithm.
#     The hyperparameters are: epochs, learning_rate, batch_size and loss.
#     The fitness function is based on the MAE of the autoencoder on the data, the lower the MAE
#     the better (higher) the fitness.
#     gene_space is a list of lists, each list is a list of possible values for each gene
#     each entry correspond to the following hyperparameters: 
#     [epochs, learning_rate, batch_size, id_loss]
#     where id_loss is an integer that corresponds to the loss function
#     0: binary_crossentropy
#     1: mse
#     2: logcosh

#     Parameters
#     ----------
#     prefix_name : str, optional
#         prefix of the name of the autoencoder, by default 'testmodel'
#     dataset : pd.DataFrame, optional
#         dataset to be encoded, by default None
#     compress_ratio : float, optional
#         compression ratio of the autoencoder, by default 0.2
#     architecture : dict, optional
#         architecture of the autoencoder, by default {'arch':'custom_VAE', 'n_factor': 0.5}
#     gene_space : list, optional
#         list of lists with the possible values for each gene, by default [list(range(10,200,20)),[0.001,0.0005],[8,16,32,64],[0,1,2]]
#     ga_settings : dict, optional
#         dictionary of settings for the genetic algorithm, by default {'num_generations': 100, 'num_parents_mating': 4, 'sol_per_pop': 10, 'parent_selection_type': 'rws', 'keep_elitism': 1}
#     savedir : str, optional
#         directory to save the results, by default './tmp/'
#     logfile : str, optional
#         name of the log file, by default 'EncoderResults.txt'
#     random_state : int, optional
#         random state for the genetic algorithm, by default 1

#     Returns
#     -------
#     pygad.gann.GANN
#         pygad object with the best hyperparameters
#     '''

#     if dataset is None:
#         raise ValueError('dataset must be provided')
#     # this is the function that I want to optimize, it is a simple autoencoder with 4 parameters
#     # epochs, learning_rate, batch_size and loss that are taken from the solution passed by pygad, 
#     # and the fitness function is the MAE of the autoencoder on the data, the lower the MAE 
#     # the better the fitness.
#     def fitness_func(solution, solution_idx):
#         # first we print the solution and the solution index
#         print(solution, solution_idx )
#         n_factor, epochs, learning_rate, batch_size, id_loss = solution
#         # only three types of loss are allowed for now, binary_crossentropy, mse and logcosh,
#         # these make sense for variational autoencoders, in particular.
#         loss={0:'binary_crossentropy',1:'mse',2:'logcosh'}[id_loss]
#         architecture['n_factor']=n_factor
#         results_dict=train_autoencoder(prefix_name = prefix_name, 
#             dataset=dataset,
#             compress_ratio = compress_ratio,
#             architecture= architecture,
#             savedir = savedir,
#             logfile = logfile, 
#             epochs = epochs,
#             learning_rate = learning_rate,
#             batch_size = batch_size, 
#             random_state = random_state,
#             loss = loss,
#             return_results = True,
#             )
#         MAE=results_dict["MAE"]
#         # if the MAE is a string, it means that the autoencoder failed to train
#         if isinstance(MAE,str):
#             MAE=1000000
#             print('ATTENTION: MAE is a string, setting it to 1000000')
#             print(f'It means that the autoencoder failed to train so solution ({solution}) is not valid.')
#         # # The fitness function calulates the sum of products between each input and its corresponding weight.
#         # output = numpy.sum(solution*function_inputs)
#         # The value 0.000001 is used to avoid the Inf value when the denominator numpy.abs(output - desired_output) is 0.0.
#         fitness = 1.0 / (MAE + 0.000001)
#         return fitness

#     # this mutation function may cause the following changes:
#     # First gene may gain or lose 0.1, but never assume value 0.
#     # Second gene may gain or lose 5, but never assume value 0.
#     # Third gene may gain or lose 0.0002, but never assume value 0.
#     # Fourth gene or Fifth gene may go the the next or previous value in the gene_space list, 
#     # it should not remain the same value.
#     def mutation_func(offspring, ga_instance):
#         for offspring_idx in range(offspring.shape[0]):
#             penalty = 0 # this is to avoid too many changes in the same individual
#             for gene_idx in range(offspring.shape[1]):
#                 # The random value should be between 0 and 1.
#                 # notice that it may or may not trigger change in other genes.
#                 # but no more than 2 genes can change due to the penalty.
#                 random_value = np.random.random() + penalty
#                 if gene_idx == 0 and random_value <= 0.5:
#                     penalty += 0.25 # max 2 genes can change
#                     random_up_or_down = np.random.random() 
#                     if random_up_or_down <= 0.5:
#                         if offspring[offspring_idx, gene_idx] + 0.1 < 3.0:
#                             offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] + 0.1
#                     else:
#                         if offspring[offspring_idx, gene_idx] - 0.1 > 0.0:
#                             offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] - 0.1
                 
#                 elif gene_idx == 1 and random_value <= 0.5:
#                     penalty += 0.25 # max 2 genes can change
#                     random_up_or_down = np.random.random() 
#                     if random_up_or_down <= 0.5:
#                         if offspring[offspring_idx, gene_idx] + 5 < 50:
#                             offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] + 5
#                     else:
#                         if offspring[offspring_idx, gene_idx] - 5 > 0:
#                             offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] - 5
#                 elif gene_idx == 2 and random_value <= 0.5:
#                     penalty += 0.25 # max 2 genes can change
#                     random_up_or_down = np.random.random() 
#                     if random_up_or_down <= 0.5:
#                         if offspring[offspring_idx, gene_idx] + 0.0002 < 1.0:
#                             offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] + 0.0002
#                     else:
#                         if offspring[offspring_idx, gene_idx] - 0.0002 > 0.0:
#                             offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] - 0.0002
#                 elif gene_idx == 3 and random_value <= 0.5:
#                     penalty += 0.25   # max 2 genes can change
#                     # It must take one of values given by the gene_space[2] but should not remain the same.
#                     offspring[offspring_idx, gene_idx] = np.random.choice(gene_space[gene_idx])
#                     ## if value is the same as initial pick another
#                     while offspring[offspring_idx, gene_idx] == ga_instance.population[offspring_idx, gene_idx]:
#                         offspring[offspring_idx, gene_idx] = np.random.choice(gene_space[gene_idx])
#                 elif gene_idx == 4 and random_value <= 0.5:
#                     penalty += 0.25  # max 2 genes can change
#                     # It must take one of values given by the gene_space[3] but should not remain the same.
#                     offspring[offspring_idx, gene_idx] = np.random.choice(gene_space[gene_idx])
#                     ## if value is the same as initial pick another
#                     while offspring[offspring_idx, gene_idx] == ga_instance.population[offspring_idx, gene_idx]:
#                         offspring[offspring_idx, gene_idx] = np.random.choice(gene_space[gene_idx])
#         return offspring
#     # last_fitness = 0 # this is giving an error
#     def on_generation(ga_instance):
#         # global last_fitness
#         print("Generation = {generation}".format(generation=ga_instance.generations_completed))
#         print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
#         # print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
#         last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
#         last_solution=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[0]
#         print("Last generation's best solutions = {solution} with fitness {fitness}.".format(solution=last_solution, fitness=last_fitness))
#         # print also the best solutions
#         print("Best solutions : ", ga_instance.best_solutions)
#         print("Best solutions fitness : ", ga_instance.best_solutions_fitness)

#     gene_type=[float, int, float, int, int]
#     # check if gene_space is a list of the types given in gene_type
#     assert len(gene_space)==len(gene_type), 'gene_space and gene_type must have the same length'
#     for i in range(len(gene_type)):
#         assert isinstance(gene_space[i],list), 'gene_space must be a list of lists'
#         assert all(isinstance(x,gene_type[i]) for x in gene_space[i]), 'gene_space must be a list of lists with the same types as gene_type'

#     ga_instance = pygad.GA(num_generations=ga_settings.pop('num_generations'),
#                         num_parents_mating=ga_settings.pop('num_parents_mating'),
#                         initial_population=initial_population,
#                         fitness_func=fitness_func,
#                         mutation_type=mutation_func,
#                         on_generation=on_generation,
#                         sol_per_pop=ga_settings.pop('sol_per_pop'),
#                         num_genes=5,
#                         gene_space=gene_space,
#                         gene_type=gene_type,
#                         allow_duplicate_genes=False, 
#                         save_best_solutions=True,
#                         save_solutions=True,
#                         parent_selection_type=ga_settings.pop('parent_selection_type'),
#                         keep_elitism=ga_settings.pop('keep_elitism'),
#                         **ga_settings
#                         )
#     # print initial population
#     print(ga_instance.population)
#     # get scores and genes of ga_instance
#     # Running the GA to optimize the parameters of the function
#     ga_instance.run()
#     ga_instance.plot_result(title="PyGAD & PyPlot - Iteration vs. Fitness", linewidth=4, save_dir='./tmp/')
#     ga_instance.save(filename=savedir+'ga_instance')
#     print("Best solutions : ", ga_instance.best_solutions)
#     print("Best solutions fitness : ", ga_instance.best_solutions_fitness)
#     # just print all solutions with a for loop with solutions and solutions_fitness
#     for i in range(len(ga_instance.solutions)):
#         print(f"Solution {i}: {ga_instance.solutions[i]}, Fitness: {ga_instance.solutions_fitness[i]}")
#     return ga_instance

# def read_past_ga(filename='tmp/ga_instance'):
#     '''
#     read past ga instance
#     Parameters
#     ----------
#     filename : str, optional
#         DESCRIPTION. The default is 'tmp/ga_instance'.
#     Returns 
#     -------
#     ga_instance : pygad.GA        
#     '''
#     ga_instance = pygad.load(filename=filename)
#     print("Best solutions : ", ga_instance.best_solutions)
#     print("Best solutions fitness : ", ga_instance.best_solutions_fitness)
#     # just print all solutions with a for loop with solutions and solutions_fitness
#     for i in range(len(ga_instance.solutions)):
#         print(f"Solution {i}: {ga_instance.solutions[i]}, Fitness: {ga_instance.solutions_fitness[i]}")
#     return ga_instance


