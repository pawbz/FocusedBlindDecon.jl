

# run this script to update the pages for Documenter.jl

# to generate doc pages
using Literate

function run_literate(names, folder)
        for t in names
                fp = joinpath(folder, string(t, ".jl"))
                output_folder=joinpath(@__DIR__, "..", "docs", "src", folder)
                println(output_folder)
                Literate.markdown(fp, output_folder, documenter=true,credit=false)
#                Literate.notebook(fp, output_folder, documenter=true,credit=false)
        end
end

run_literate(["tut1", "tut2"], "")

