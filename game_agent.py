"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

infinity = float('inf')

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """


    return number_moves_heuristic(game, player)

def number_moves_heuristic(game, player):

        x = 3

        if game.is_loser(player):
            return -infinity

        if game.is_winner(player):
            return infinity

        # players moves
        moves = game.get_legal_moves(player)
        moves_count = len(moves)
        # oponents moves
        opponent_moves = game.get_legal_moves(game.get_opponent(player))
        opponent_moves_count = len(opponent_moves)

        # intersect the legal moves of my player and opponent
        # if the resulting list is of length 0, there is an intersection
        # at this point the player with the most moves will win
        # until this point, i will use moves - opponent_moves
        intersected_lists = [val for val in moves if val in opponent_moves]

        if not len(intersected_lists):
            return float(moves_count)
        else:
            return float(moves_count - x * opponent_moves_count)

def minimize_opponent_heuristic(game, player):

        if game.is_loser(player):
            return -infinity

        if game.is_winner(player):
            return infinity

        # oponents moves
        opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

        # garantee the least amount of opponent
        return float(-opponent_moves)


def filled_spaces_heuristic(game, player):
    # x scales the weight of opponent moves
    x = 3
    board_width = 7
    board_height = 7

    if game.is_loser(player):
        return -infinity

    if game.is_winner(player):
        return infinity

    # players moves
    moves = len(game.get_legal_moves(player))
    # oponents moves
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    board_tiles = board_width * board_height
    filled_spaces = board_tiles - len(game.get_blank_spaces())
    return float((moves - (x * opponent_moves)) * (filled_spaces))

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return(-1,-1)

        best_move = legal_moves[0]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # iterative deepening
            if self.iterative:
                # start on the first level
                depth = 1;
                while True:
                    if self.method == "minimax":
                        # only care about best move, calling minimax on every level
                        _, best_move = self.minimax(game, depth)
                    if self.method == "alphabeta":
                        # only care about best move, calling alphabeta on every level
                        _, best_move = self.alphabeta(game, depth)

                    # next level
                    depth += 1

            # fixed
            if self.method == "minimax":
                # call minimax on a set depth
                _, best_move = self.minimax(game, self.search_depth)
            if self.method == "alphabeta":
                # call alphabeta on a set depth
                _, best_move = self.alphabeta(game, self.search_depth)

        except Timeout:
            # return the current best move if time runs out
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # available moves to player
        legal_moves = game.get_legal_moves()

        # worst case
        if not len(legal_moves):
            return (self.score(game, self), (-1, -1))

        # grab first location as starting move
        current_move = legal_moves[0]

        # set upper bound
        current_score = infinity
        if maximizing_player:
            # maximizing is looking for score greater than negative infinity
            # set lower bound
            current_score = -infinity

        # worst case
        if not len(legal_moves):
            return (self.score(game, self), (-1, -1))

        # begin the traversal if depth is > 0
        if depth:
            for legal_move in legal_moves:
                new_game = game.forecast_move(legal_move)

                # recursively call as the oposite player (toggle between min/max)
                if maximizing_player:
                    score, _ = self.minimax(new_game, depth-1, False)
                else:
                    score, _ = self.minimax(new_game, depth-1, True)

                # if maximizing player finds a score greater than the current score
                # set current_score to the new found sccore and update the current_move to
                # the current legal move.
                # Do the same for the minimizing player, but only if the score is less
                if maximizing_player and score > current_score:
                    current_score = score
                    current_move = legal_move
                elif not maximizing_player and score < current_score:
                    current_score = score
                    current_move = legal_move
            return (current_score, current_move)

        # if all else
        return (self.score(game, self), (-1, -1))

    def alphabeta(self, game, depth, alpha=-infinity, beta=infinity, maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        # worst case
        if not len(legal_moves):
            return (self.score(game, self), (-1, -1))

        # init current move as first move in legal moves
        current_move = legal_moves[0]

        # set upper bound
        current_score = infinity
        if maximizing_player:
            # maximizing is looking for score greater than negative infinity
            # set lower bound
            current_score = -infinity

        if depth:
            for legal_move in legal_moves:
                # base case, if alpha is greater than beta, we can ignore the
                # remaining nodes for that tree
                if alpha >= beta:
                    return (current_score, current_move)

                new_game = game.forecast_move(legal_move)

                # recursively call alphabeta on oposite players, passing the current
                # alpha and beta values
                if maximizing_player:
                    score, _ = self.alphabeta(new_game, depth-1, alpha, beta, False)
                else:
                    score, _ = self.alphabeta(new_game, depth-1, alpha, beta, True)

                if maximizing_player and score > current_score:
                    # same as minimax, but update alpha for max player
                    alpha = score
                    current_score = score
                    current_move = legal_move
                elif not maximizing_player and score < current_score:
                    # same as minimax, but update beta for min player
                    beta = score
                    current_score = score;
                    current_move = legal_move;

            # returns the alpha for max and beta for min
            return (current_score, current_move)

        # if all else fails
        return (self.score(game, self), (-1, -1))
