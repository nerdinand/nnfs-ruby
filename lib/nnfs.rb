require 'numo/narray'

module NNFS
  def init(random_seed: 0)
    Numo::NArray.srand(random_seed)
  end
end
