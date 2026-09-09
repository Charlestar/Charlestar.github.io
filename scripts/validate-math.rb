# frozen_string_literal: true

# Check the actual Jekyll output, not a second rendering of the source.
# Kramdown's native math nodes preserve TeX before Markdown consumes escapes.
require "jekyll"
require "cgi"
require "kramdown"
require "kramdown-parser-gfm"

config = Jekyll.configuration("destination" => ARGV.fetch(0, "_site"))
site = Jekyll::Site.new(config)
site.reset
site.read

def math_nodes(node, result = [])
  if node.type == :math
    result << [node.options[:category], node.value.gsub(/\s+/, " ").strip]
  end
  node.children.each { |child| math_nodes(child, result) }
  result
end

failures = []
inline_count = 0
display_count = 0
checked_posts = 0

site.posts.docs.each do |post|
  next unless post.data["mathjax"]

  options = site.config.fetch("kramdown", {}).transform_keys(&:to_sym)
  document = Kramdown::Document.new(post.content, **options)
  expected = math_nodes(document.root)
  next if expected.empty?

  output_path = post.destination(site.dest)
  unless File.file?(output_path)
    failures << "#{post.relative_path}: missing built page #{output_path}"
    next
  end

  # Exclude verbatim examples: TeX in code is not evidence of rendered prose.
  html = File.read(output_path, encoding: "UTF-8")
  html = html.gsub(/<(pre|code)\b[^>]*>.*?<\/\1>/m, "")
  available = Hash.new(0)
  html.scan(/\\\((.*?)\\\)|\\\[(.*?)\\\]/m) do |inline, display|
    category, formula = inline ? [:span, inline] : [:block, display]
    key = [category, CGI.unescapeHTML(formula).gsub(/\s+/, " ").strip]
    available[key] += 1
  end

  expected.each do |category, formula|
    category == :span ? inline_count += 1 : display_count += 1
    key = [category, formula]
    if available[key].positive?
      available[key] -= 1
    else
      failures << "#{post.relative_path}: missing or altered #{category} math: #{formula[0, 160]}"
    end
  end
  checked_posts += 1
end

abort failures.join("\n") unless failures.empty?
puts "Validated #{inline_count} native inline and #{display_count} display formulas in #{checked_posts} built posts."
puts "Single-dollar legacy math and browser typesetting require separate checks."
