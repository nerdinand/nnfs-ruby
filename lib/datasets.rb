require 'numo/narray'
require 'gruff'

class Plot
  def plot_2d(x, y, file_name)
    g = Gruff::Scatter.new

    y.to_a.uniq.each do |klass|
      xs = x[y.eq(klass).where, true]
      g.data(klass, xs[true, 0], xs[true, 1])
    end

    g.write(file_name)
  end
end


class VerticalData
	def self.create(n_samples:, n_classes:)
    x = Numo::DFloat.zeros(n_samples * n_classes, 2)
    y = Numo::Int8.zeros(n_samples * n_classes)

    (0...n_classes).each do |class_number|
      ix = ((n_samples * class_number)...(n_samples * (class_number + 1)))
      x[ix, true] = Numo::NArray.column_stack([Numo::DFloat.new(n_samples).rand_norm * 0.1 + class_number / 3.0, Numo::DFloat.new(n_samples).rand_norm * 0.1 + 0.5])
      y[ix] = class_number
    end

    [x, y]
	end
end

class SineData
  def self.create(n_samples:)
    x = Numo::DFloat.linspace(-1, 1, n_samples)
    y = Numo::DFloat::Math.sin(2 * Math::PI * x)
    [Numo::NArray[x, y].transpose, Numo::Int8.zeros(n_samples)]
  end
end

class SpiralData
  def self.create(n_samples:, n_classes:)
    x = Numo::DFloat.zeros(n_samples * n_classes, 2)
    y = Numo::Int8.zeros(n_samples * n_classes)

    (0...n_classes).each do |class_number|
      ix = ((n_samples * class_number)...(n_samples * (class_number + 1)))

      r = Numo::DFloat.linspace(0, 1, n_samples)
      t = Numo::DFloat.linspace(class_number * 4, (class_number + 1) * 4, n_samples) + Numo::DFloat.new(n_samples).rand_norm * 0.2
      x[ix, true] = Numo::NArray.column_stack([r * Numo::DFloat::Math.sin(t * 2.5), r * Numo::DFloat::Math.cos(t * 2.5)])
      y[ix] = class_number
    end

    [x, y]
  end
end