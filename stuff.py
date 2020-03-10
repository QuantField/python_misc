

class myrange:

    def __init__(self, start=1, end=1):
        self.start = start
        self.end   = end
        self.val   = self.start-1

    def __iter__(self):        
        return self
        
    def __next__(self):        
        if self.val> self.end-1:
            raise StopIteration
        else:            
            self.val+=1
            return self.val 
       


def myrange2(start, end):
    val = start
    while val<=end:
        yield val
        val += 1



for r in myrange(1,5):
    print(r)

print()

for r in myrange2(1,5):
    print(r)














