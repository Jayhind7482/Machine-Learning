import numpy as np
class KMeans():
    def __init__ (self , max_iter = 100 , n_cluster=4):
        self.max_iter = max_iter
        self.n_cluster = n_cluster 
        self.centroids = None 
        
    # Assign Random centroids value for each clsuter from X
    # calculate the cluster number for each X on the basis of euclidean distance from centoid
    # move the cluster 
    # finsish if max_iter is over or cluster position is not changing 
    
    
    def fit_predict(self , X):
        # Initialization of centroid
        np.random.seed(42)
        tem_cend = []
        for i in range(self.n_cluster):
            rand_index = np.random.randint(0 , len(X) )
            tem_cend.append(X[rand_index])
        self.centroids = np.array(tem_cend)  # Intialization is done 
        for i in range(self.max_iter):
            # assigining the clsuter to each X
            clsuter_group = self.assign_clsuter( X )
            old_centroids = self.centroids
            # Moving the clsuter 
            self.centroids = self.move_clsuter (X , clsuter_group)
            if (old_centroids==self.centroids).all():
                break
        return clsuter_group
        
        
    def assign_clsuter(self,X):
        group = []
        for j in X:
            distance = []
            for k in self.centroids:
                distance.append(np.sqrt(np.dot(j-k , j-k)))
            index = np.argmin(np.array(distance))
            group.append(index)
        return np.array(group)
    
    def move_clsuter(self , X , clsuter_group):
        new_centroids = []
        clsuter_type = np.unique(clsuter_group)
        for i in clsuter_type:
            points = X[clsuter_group==i]
            if len(points) > 0:
            # mean of assigned points
                new_centroids.append(points.mean(axis=0))
            else:
                # if no points in this cluster, reinitialize randomly
                rand_index = np.random.randint(0, len(X))
                new_centroids.append(X[rand_index])
        
        return np.array(new_centroids)
    
        
            
            

            
        
            
        