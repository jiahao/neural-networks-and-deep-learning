using Test
using Random
using JLD2, FileIO

Random.seed!(0)

let
    include("../src/run-ch1.jl")

    @test net == net_test
    @test evaluate(net, test_data) == score_test
end
