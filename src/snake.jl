using SnakeGames
using ReinforcementLearning
mutable struct SnakeGameEnv{A,N,G} <: AbstractEnv
    game::G
    latest_snakes_length::Vector{Int}
    latest_actions::Vector{CartesianIndex{2}}
    is_terminated::Bool
end

function SnakeGameEnv(; action_style=MINIMAL_ACTION_SET, kw...)
    game = SnakeGame(; kw...)
    n_snakes = length(game.snakes)
    num_agent_style = n_snakes == 1 ? SINGLE_AGENT : MultiAgent{n_snakes}()
    SnakeGameEnv{action_style,num_agent_style,typeof(game)}(
        game,
        map(length, game.snakes),
        Vector{CartesianIndex{2}}(undef, length(game.snakes)),
        false,
        )
    end

SnakeGameEnv()
RLBase.ActionStyle(env::SnakeGameEnv{A}) where {A} = A
RLBase.NumAgentStyle(env::SnakeGameEnv{<:Any,N}) where {N} = N
RLBase.DynamicStyle(env::SnakeGameEnv{<:Any,SINGLE_AGENT}) = SEQUENTIAL
RLBase.DynamicStyle(env::SnakeGameEnv{<:Any,<:MultiAgent}) = SIMULTANEOUS

const SNAKE_GAME_ACTIONS = (
    CartesianIndex(-1, 0),
    CartesianIndex(1, 0),
    CartesianIndex(0, 1),
    CartesianIndex(0, -1),
)

# function RLBase.act!(env::SnakeGameEnv{A}, actions::Vector{CartesianIndex{2}}) where {A}
#     if A === MINIMAL_ACTION_SET
#         # avoid turn back
#         actions = [
#             a_new == -a_old ? a_old : a_new for
#             (a_new, a_old) in zip(actions, env.latest_actions)
#         ]
#     end

#     env.latest_actions .= actions
#     map!(length, env.latest_snakes_length, env.game.snakes)
#     env.is_terminated = !env.game(actions)
# end

# RLBase.act!(env::SnakeGameEnv, action::Int) = env([SNAKE_GAME_ACTIONS[action]])
# RLBase.act!(env::SnakeGameEnv, actions::Vector{Int}) = env(map(a -> SNAKE_GAME_ACTIONS[a], actions))

function (env::SnakeGameEnv{A})(actions::Vector{CartesianIndex{2}}) where {A}
    if A === MINIMAL_ACTION_SET
        # avoid turn back
        actions = [
            a_new == -a_old ? a_old : a_new for
            (a_new, a_old) in zip(actions, env.latest_actions)
        ]
    end

    env.latest_actions .= actions
    map!(length, env.latest_snakes_length, env.game.snakes)
    env.is_terminated = !env.game(actions)
end

(env::SnakeGameEnv)(action::Int) = env([SNAKE_GAME_ACTIONS[action]])
(env::SnakeGameEnv)(actions::Vector{Int}) = env(map(a -> SNAKE_GAME_ACTIONS[a], actions))


RLBase.action_space(env::SnakeGameEnv) = Base.OneTo(4)
#RLBase.state(env::SnakeGameEnv) = env.game.board
#RLBase.state_space(env::SnakeGameEnv) = ArrayProductDomain(fill(false:true, size(env.game.board)))
#RLBase.reward(env::SnakeGameEnv{<:Any,SINGLE_AGENT}) =
#    length(env.game.snakes[]) - env.latest_snakes_length[]
#RLBase.reward(env::SnakeGameEnv) = length.(env.game.snakes) .- env.latest_snakes_length
RLBase.is_terminated(env::SnakeGameEnv) = env.is_terminated

# This only works with a single snake and one food at a time
function RLBase.state(env::SnakeGameEnv)

    game = env.game
    snake = game.snakes[]
    food = game.foods[]
    walls = game.walls

    snake_left = snake[1] + CartesianIndex(-1, 0)
    snake_right = snake[1] + CartesianIndex(1, 0)
    snake_up = snake[1] + CartesianIndex(0, -1)
    snake_down = snake[1] + CartesianIndex(0, 1)

    prev_action = env.latest_actions[]

    return [
        # Snake's position relative to the food
        food[1] < snake[1], # Food is to the left of the snake
        food[1] > snake[1], # Food is to the right of the snake
        food[2] < snake[2], # Food is above the snake
        food[2] > snake[2], # Food is below the snake

        # Snake's position relative to danger
        snake_left ∈ snake || snake_left ∈ walls, # Obstacle directly to the left of the snake
        snake_right ∈ snake || snake_right ∈ walls, # Obstacle directly to the right of the snake
        snake_up ∈ snake || snake_up ∈ walls, # Obstacle directly above the snake
        snake_down ∈ snake || snake_down ∈ walls, # Obstacle directly below the snake

        # Snake's direction
        prev_action[1] < 0, # Snake going left
        prev_action[1] > 0, # Snake going right
        prev_action[2] < 0, # Snake going up
        prev_action[2] > 0, # Snake going down
    ]

end

RLBase.state_space(env::SnakeGameEnv) = ArrayProductDomain(fill(false:true, size(RLBase.state(env))))

# This only works with a single snake and one food at a time
function RLBase.reward(env::SnakeGameEnv)

    game = env.game
    snake = game.snakes[]
    food = game.foods[]
    walls = game.walls

    prev_action = env.latest_actions[]
    prev_pos = CartesianIndex(mod.((snake[1] - prev_action).I, axes(game.board)[1:end-1]))

    dist = (p1, p2) -> hypot(p2[1] - p1[1], p2[2] - p1[2])

    # First check if food was consumed
    if length(snake) > env.latest_snakes_length[]
        reward = 10
    elseif env.is_terminated
        reward = -100
    elseif dist(snake[1], food) < dist(prev_pos, food)
        reward = 1
    elseif dist(snake[1], food) > dist(prev_pos, food)
        reward = -1
    else
        reward = 0
    end

    return reward

end

RLBase.legal_action_space(env::SnakeGameEnv{FULL_ACTION_SET,SINGLE_AGENT}) =
    findall(!=(-env.latest_actions[]), SNAKE_GAME_ACTIONS)
RLBase.legal_action_space(env::SnakeGameEnv{FULL_ACTION_SET}) =
    [findall(!=(-a), SNAKE_GAME_ACTIONS) for a in env.latest_actions]

RLBase.legal_action_space_mask(env::SnakeGameEnv{FULL_ACTION_SET,SINGLE_AGENT}) =
    [a != -env.latest_actions[] for a in SNAKE_GAME_ACTIONS]
RLBase.legal_action_space_mask(env::SnakeGameEnv{FULL_ACTION_SET}) =
    [[x != -a for x in SNAKE_GAME_ACTIONS] for a in env.latest_actions]

function RLBase.reset!(env::SnakeGameEnv)
    SnakeGames.reset!(env.game)
    env.is_terminated = false
    fill!(env.latest_actions, CartesianIndex(0, 0))
    map!(length, env.latest_snakes_length, env.game.snakes)
end

Base.display(io::IO, m::MIME, env::SnakeGameEnv) = display(io, m, env.game)

# function new_func(scene, game, actions)
#     record(scene, f_name; framerate=framerate) do io
#         for action in actions
#             is_success = game(action)
#             game_node[] = game
#             recordframe!(io)
#             is_success || break
#             is_exit[] && break
#         end
#     end
# end

# Custom hook
# https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent-Tuple{AbstractStage,%20AbstractEnv}
# (hook::YourHook)(::PostActStage, agent, env)
# (hook::YourHook)(::PreEpisodeStage, agent, env)
