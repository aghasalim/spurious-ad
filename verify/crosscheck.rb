# Cross file identities that hold if the runs and the tables are consistent,
# and that nothing in the repository checked.
#
# The recomputations elsewhere in verify/ ask whether each published table is
# the right aggregate of its own raw file. This asks three different questions:
#
#   balance      does every cell hold the full 5 categories x 3 seeds grid? An
#                unbalanced cell would still produce a mean, just not the mean
#                the README describes.
#   folding      the per detector summary and the per category table are two
#                aggregates of the same runs, so with balanced cells the
#                category table must average back to the summary. A mismatch
#                means one of the two groupbys read different rows. Both sides
#                are already rounded to four decimals, so this one is held to
#                twice the usual tolerance.
#   determinism  experiments/real.py --mode sweep and --mode mechanism share a
#                configuration at pin off, and the synthetic sweep and mechanism
#                runs share one too. Same seeds, same generator, no sampling
#                left over, so those rows must be identical value for value.
#                They are written by separate invocations hours apart, so this
#                is a real check that the pipeline is reproducible.
#
# Run: ruby verify/crosscheck.rb <repo root>

require 'json'
require 'csv'

ROOT = ARGV[0] || '.'
TOL = 6e-5          # half of the last published decimal place, plus headroom
FOLD_TOL = 1.2e-4   # folding compares two tables each rounded to four decimals
EXACT = 1e-12       # rows from the same configuration should be bit identical

$failures = 0

def fail(msg)
  puts "  FAIL #{msg}"
  $failures += 1
end

def load_json(name)
  JSON.parse(File.read(File.join(ROOT, 'reports', name)))
end

def load_csv(name)
  CSV.read(File.join(ROOT, 'reports', name), headers: true).map(&:to_h)
end

def fmt(v)
  return 'True' if v == true
  return 'False' if v == false
  v.is_a?(Float) ? (v == v.round ? v.round.to_s : v.to_s) : v.to_s
end

def key(row, fields)
  fields.map { |f| fmt(row[f]) }.join('|')
end

def csv_key(row, fields)
  fields.map { |f|
    v = row[f]
    v =~ /\A-?\d+(\.\d+)?\z/ ? fmt(Float(v)) : v
  }.join('|')
end

def mean(a)
  a.reduce(:+) / a.size.to_f
end

# --- balance -----------------------------------------------------------------

puts 'cell balance in the raw runs'
{
  'real_sweep.json'     => %w[category detector rho],
  'real_backbone.json'  => %w[category detector rho],
  'real_mechanism.json' => %w[category detector rho pinned_train_rate],
  'sweep.json'          => %w[rho],
  'mechanism.json'      => %w[rho pinned_train_rate]
}.each do |name, fields|
  rows = load_json(name)
  cells = rows.group_by { |r| key(r, fields) }
  seeds = cells.values.map { |g| g.map { |r| r['seed'] }.sort }.uniq
  cats = rows.group_by { |r| key(r, fields - %w[category]) }
             .values.map { |g| g.map { |r| r['category'] }.uniq.sort }.uniq
  if seeds != [[0, 1, 2]]
    fail("#{name}: cells do not all hold seeds 0, 1, 2, saw #{seeds.inspect}")
  elsif cats.size != 1
    fail("#{name}: cells do not all hold the same categories, saw #{cats.inspect}")
  elsif fields.include?('category')
    puts format('  %-22s %3d runs in %2d cells, every cell 3 seeds x %d categories',
                name, rows.size, cells.size, cats.first.size)
  else
    puts format('  %-22s %3d runs in %2d cells, every cell 3 seeds',
                name, rows.size, cells.size)
  end
end

# --- folding -----------------------------------------------------------------

puts
puts 'the per category table averages back to the per detector summary'
METRICS = %w[auroc car car_random peak_on_defect].freeze

%w[sweep mechanism backbone].each do |mode|
  by_cat = load_csv("real_#{mode}_by_category.csv")
  summ = load_csv("real_#{mode}_summary.csv")
  worst = 0.0
  cells = 0

  by_cat.group_by { |r| csv_key(r, %w[detector rho]) }.each do |k, group|
    # The mechanism summary splits on the pin while its category table does
    # not, so the two pin rows, which hold the same number of runs, average to
    # the category table's value.
    want_rows = summ.select { |r| csv_key(r, %w[detector rho]) == k }
    if want_rows.empty?
      fail("real_#{mode}: #{k} is in the category table but not the summary")
      next
    end
    METRICS.each do |m|
      got = mean(group.map { |r| Float(r[m]) })
      want = mean(want_rows.map { |r| Float(r[m]) })
      d = (got - want).abs
      worst = d if d > worst
      cells += 1
      fail("real_#{mode} #{k} #{m}: category table #{got}, summary #{want}") if d > FOLD_TOL
    end
  end
  puts format('  %-24s %3d cells  max |d| %.2e', "real_#{mode}", cells, worst)
end

puts
puts 'the summary n column is the sum of the per run n'
%w[sweep mechanism backbone].each do |mode|
  runs = load_json("real_#{mode}.json")
  keys = mode == 'mechanism' ? %w[detector pinned_train_rate rho] : %w[detector rho]
  summ = load_csv("real_#{mode}_summary.csv")
  bad = 0
  summ.each do |row|
    k = csv_key(row, keys)
    got = runs.select { |r| key(r, keys) == k }.map { |r| r['n'] }.reduce(0, :+)
    want = Integer(row['n'])
    if got != want
      fail("real_#{mode} #{k} n: raw sum #{got}, published #{want}")
      bad += 1
    end
  end
  puts format('  %-24s %2d rows, all n match exactly', "real_#{mode}", summ.size) if bad.zero?
end

# --- determinism across separate invocations ---------------------------------

puts
puts 'runs that share a configuration across two invocations are identical'

def compare_shared(name_a, rows_a, name_b, rows_b, keys, skip)
  shared = (rows_a.first.keys & rows_b.first.keys) - keys - skip
  matched = 0
  worst = 0.0
  rows_a.each do |a|
    b = rows_b.find { |r| keys.all? { |k| r[k] == a[k] } }
    next if b.nil?
    matched += 1
    shared.each do |f|
      next unless a[f].is_a?(Numeric)
      d = (a[f] - b[f]).abs
      worst = d if d > worst
      fail("#{name_a} vs #{name_b} #{keys.map { |k| a[k] }.join(',')} #{f}: " \
           "#{a[f]} against #{b[f]}") if d > EXACT
    end
  end
  [matched, shared.size, worst]
end

sweep = load_json('real_sweep.json')
mech_free = load_json('real_mechanism.json').reject { |r| r['pinned_train_rate'] }
m, f, w = compare_shared('real_sweep.json', mech_free, 'real_mechanism.json', sweep,
                         %w[category detector rho seed], %w[pinned_train_rate])
if m != mech_free.size
  fail("only #{m} of #{mech_free.size} unpinned mechanism runs found in the sweep")
else
  puts format('  real_mechanism (pin off) vs real_sweep   %3d runs x %d fields  max |d| %.1e',
              m, f, w)
end

syn_sweep = load_json('sweep.json')
syn_mech_free = load_json('mechanism.json').reject { |r| r['pinned_train_rate'] }
m, f, w = compare_shared('mechanism.json', syn_mech_free, 'sweep.json', syn_sweep,
                         %w[rho seed], %w[pinned_train_rate train_confound_rate])
if m != syn_mech_free.size
  fail("only #{m} of #{syn_mech_free.size} unpinned synthetic mechanism runs " \
       'found in the sweep')
else
  puts format('  mechanism (pin off) vs sweep             %3d runs x %d fields  max |d| %.1e',
              m, f, w)
end

puts
if $failures > 0
  puts "#{$failures} cross file checks failed"
  exit 1
end
puts 'Ruby finds the cells balanced, the two groupbys folding into each other,'
puts 'and the shared configurations reproducing exactly'
