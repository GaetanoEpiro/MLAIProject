import itertools 
import random 

from time import time 

def hamming_distance(string1, string2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance


def main(): 
  start = time()
  
  n_permutations = 30
  dimension = 9
  
  initial = list(range(dimension))
  list_permutations = list(itertools.permutations(initial))
  
  # take out the correct permutation because it will be used in the Dataset
  first_tuple = tuple(range(dimension))
  list_permutations.remove(first_tuple)
  print("Total permutations: ", len(list_permutations))

  # shuffle the list of permutations
  random.shuffle(list_permutations)

  #list of the final permutations
  final_permutations = []   

  #take the first final permutation as a random
  first_random = random.choice(list_permutations)
  final_permutations.append(first_random)

  #start with the maximum distance possible 
  max_distance = 9

  min_distance_taken = 9

  while len(final_permutations) <30: 
    

    for perm in list_permutations: 
      dist_count = 0

      for good in final_permutations: 
        dist = hamming_distance(good, perm)

        if dist == dimension: 
          dist_count += 1
        elif dist>= max_distance: 
          dist_count += 1
          if dist < min_distance_taken: 
            min_distance_taken = dist
        
      if dist_count == len(final_permutations): 
        #perm is a good permutation, should be added to final_permutation
        if len(final_permutations) <30 : 
          final_permutations.append(perm)
          list_permutations.remove(perm)
          print("Permutation  " + str(len(final_permutations)) + ":    " +  str(perm))
        else: 
          break
      if len(final_permutations) >= 30: 
        break
    
    #no permutations found that have max_distance, reduce max_distance
    max_distance = max_distance -1
  


  print("Permutations found:   ", final_permutations)
  print("Total time taken: ", time() - start)
  print("Minimum distance taken: ", min_distance_taken)


  with open('permutationshamming%d.txt' %(n_permutations), 'w') as f:
    for i in range(0, len(final_permutations)):
      f.write(' '.join(str(n) for n in final_permutations[i])[:] + '\n')

if __name__ == "__main__":
    main()