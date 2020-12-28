require 'numo/narray'

class NNFS
  def self.init(random_seed: 0)
    Numo::NArray.srand(random_seed)
  end
end
