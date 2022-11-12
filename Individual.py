import random
class Individual():
    
    length_of_genes = 30
    @classmethod
    def mutate_genes(self):
        gene = random.choice("0123") 
        return gene

    @classmethod
    def create_1_gene(self,len = 100):
       
        self.length_of_genes = len
        return [self.mutate_genes() for _ in range(self.length_of_genes)]

    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def cross_and_mutate(self, par2, generation):
        child_chromosome = []
    
        c = random.randint(0,self.length_of_genes) 

        prob = random.random()
        child_chromosome = self.chromosome[:c] + par2.chromosome[c:] 
        mutation_prob = 0.8 if generation > 100 else 0.6 if generation > 50 else 0.4
        if prob < mutation_prob:
            leng_muta = self.length_of_genes
            a = random.randrange(0,leng_muta)
            for _ in range(a):
                p = random.randrange(0,self.length_of_genes)
                child_chromosome[p]= self.mutate_genes()
                
        return Individual(child_chromosome)