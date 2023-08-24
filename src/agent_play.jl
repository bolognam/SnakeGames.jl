function play(agent::Agent, env; f_name = "agent_game.mp4", framerate = 10)

    (scene, game_node) = scene_and_node(env.game)

    record(scene, f_name; framerate) do io
        run(agent.policy, env, StopWhenDone(), DoEveryNStep() do t, p, e
            sleep(0.1)
            game_node[] = e.game
            recordframe!(io)
        end)
    end

    println("game over")

end
