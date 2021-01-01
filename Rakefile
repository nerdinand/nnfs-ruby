task default: %w[test]

task :test do
  Dir['*/*.rb'].grep(/\/\d\d.+\.rb$/).sort.each do |file_path|
    puts "Running #{file_path}..."
    ruby file_path
  end
end
