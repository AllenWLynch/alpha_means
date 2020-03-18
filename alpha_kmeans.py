
import numpy as np

class AlphaKmeans():

    def __init__(self, k):
        self.k = k
        self.initialized = False
        self.optimized = False

    def _optimal_lambda(self, alphas):
        alphabar = np.mean(alphas)
        return (3/4) * (np.exp(np.log(alphabar/self.optimal_workload)/(alphabar - self.optimal_workload)) - 1)

    def _centroid_distances(self, data):
        return np.sum(np.square(data[:,np.newaxis,:] - self.centers[np.newaxis, :, :]), axis = -1)

    def _optimize(self, data, alphas):
        self.cluster_labels = np.zeros(self.N)

        self.alpha_costs = np.zeros(self.k)
        self.overall_cost = np.zeros(self.k)

        distances = self._centroid_distances(data)

        for i in self._sample_points(alphas):
            alphai = alphas[i]
            alphamod = alphai * (1 + self.lambd)**(self.alpha_costs - self.optimal_workload + alphai)
            cost = distances[i] + alphamod

            min_cost, assignment = np.min(cost), np.argmin(cost)
            self.alpha_costs[assignment] += alphamod[assignment]
            self.cluster_labels[i] = assignment

        self._update_centers(data)

    def _update_centers(self, data):
        for k in range(self.k):
            substituents = data[self.cluster_labels == k]
            #if substituents.size() > 0:
            self.centers[k] = substituents.mean(axis = 0)

    def _sample_points(self, alphas):
        inverse_size = 1/alphas
        inverse_size_proportional = np.squeeze(inverse_size/inverse_size.sum())
        return np.random.choice(np.arange(self.N), self.N, p = inverse_size_proportional, replace = False)

    def _initialize(self, data, alphas, lambd = None):
        #convert to tensor for parallelization
        assert(len(alphas.shape) <= 2), 'Node weights should be (N,1) or (N,) shaped tensor'
        assert(data.shape[0] == alphas.shape[0]), 'Data and weights should have same length'

        N, d = data.shape
        self.N = N

        alphas = alphas.reshape((-1,))

        self.optimal_workload = np.sum(alphas)/self.k

        self.cluster_labels = np.random.randint(0, self.k, N)
        
        if lambd is None:
            lambd = self._optimal_lambda(alphas)
        self.lambd = lambd
    
        self.centers = np.zeros((self.k, d))
        self.centers[0] = data[np.random.choice(N, 1)]
        for i in range(1, self.k):
            min_distances = np.reshape(self._centroid_distances(data)[:, i], (N,1)).min(axis = -1)
            probs = min_distances/np.sum(min_distances)
            selection = np.random.choice(np.arange(N), p = probs)
            self.centers[i] = data[selection]
        
        self.initialized = True

    def transform(self, data):
        return np.min(self._centroid_distances(data), axis = -1)
        
    def fit(self, data, alphas, lambd = None, max_iters = 50, verbose = True, metrics = False):

        if verbose:
            print('Initializing centers ...')

        self._initialize(data, alphas, lambd = lambd)

        if verbose:
            print('Optimizing')
        prev_labels = self.cluster_labels
        for i in range(max_iters):
            if verbose:
                print('\rIteration: {}'.format(str(i+1)), end = '')
            self._optimize(data, alphas)
            if np.sum(self.cluster_labels - prev_labels) == 0:
                if verbose:
                    print('\nStable configuration reached.',end ='')
                break
        if verbose:
            print('\nComplete')

        self.optimized = False

        if metrics:
            return self.cluster_labels, self.get_metrics(data, alphas)
        else:
            return self.cluster_labels

    def get_centers(self):
        return self.centers

    @staticmethod
    def static_get_cluster_metrics(distances, alphas, labels):
        cluster_metrics = []
        for k in range(int(labels.max())):
            subs = labels == k
            distance_cost = distances[subs][:,k].sum()
            alpha_cost = alphas[subs].sum()
            cluster_metrics.append((distance_cost, alpha_cost, distance_cost+alpha_cost))
        return {name: l for name, l in zip(['cluster_distance', 'cluster_weight','cluster_total'],list(zip(*cluster_metrics)))}
        #return list(zip(*cluster_metrics))

    def get_metrics(self, data, alphas):
        return self.static_get_cluster_metrics(self._centroid_distances(data), alphas, self.cluster_labels)

    def get_max_cost(self, data, alphas):
        return max(self.get_metrics(data, alphas)['cluster_total'])