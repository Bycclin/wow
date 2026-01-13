#!/usr/bin/env python3
# -- coding: utf-8 --
import argparse
import asyncio
import chess
import chess.engine
import chess.polyglot
import chess.pgn
import logging
import random
import copy
import subprocess


def print_unicode_board(board, perspective=chess.WHITE):
    sc, ec = '\x1b[0;30;107m', '\x1b[0m'
    
    for r_loop in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line_rank_label = r_loop + 1
        line = [f"{sc} {line_rank_label}"]

        for c_loop in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = '\x1b[48;5;255m' if (r_loop + c_loop) % 2 == 1 else '\x1b[48;5;253m'
            current_square_index = chess.square(c_loop, r_loop)

            if board.move_stack:
                last_move = board.move_stack[-1]
                if last_move.to_square == current_square_index or last_move.from_square == current_square_index:
                    color = '\x1b[48;5;153m'
            
            piece = board.piece_at(current_square_index)
            symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol()] if piece else ' '
            line.append(color + symbol)

        print(" " + " ".join(line) + f" {sc}{ec}")

    if perspective == chess.WHITE:
        print(f" {sc}   a b c d e f g h  {ec}\n")
    else:
        print(f" {sc}   h g f e d c b a  {ec}\n")

async def get_human_move(board):
    loop = asyncio.get_running_loop()
    while True:
        try:
            move_str = await loop.run_in_executor(None, input, "Your move (SAN or UCI): ")
            move_str = move_str.strip()
            
            try:
                move = board.parse_san(move_str)
            except ValueError:
                move = board.parse_uci(move_str)
            
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move. Please try again.")
        except ValueError:
            print("Invalid format. Use standard notation (e.g., e4, Nf3) or UCI (e.g., e2e4).")


async def load_engine(engine_command, engine_options=None, debug=False):
    cmd = engine_command.split()
    popen_args = {}
    if not debug:
        popen_args["stderr"] = subprocess.DEVNULL

    transport = None
    engine = None
    protocol_used = None

    try:
        transport, engine = await asyncio.wait_for(
            chess.engine.popen_uci(cmd, **popen_args), timeout=5
        )
        protocol_used = 'uci'
    except Exception:
        try:
            transport, engine = await asyncio.wait_for(
                chess.engine.popen_xboard(cmd, **popen_args), timeout=5
            )
            protocol_used = 'xboard'
        except Exception:
            raise RuntimeError(f"Engine '{engine_command}' failed to initialize (tried UCI, then CECP/XBoard).")

    if hasattr(engine, "debug") and callable(engine.debug):
        if asyncio.iscoroutinefunction(engine.debug):
            await engine.debug(debug)
        else:
            engine.debug(debug) 
    
    if engine_options:
        try:
            if protocol_used == 'uci' and 'Chess960' in engine.options:
                is_chess960_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" not in engine.board.fen()
                if is_chess960_board:
                    engine_options['Chess960'] = True
            await engine.configure(engine_options)
        except Exception as e:
            await engine.quit()
            raise RuntimeError(f"Failed to configure engine '{engine_command}' with options {engine_options}: {e}")
    
    return engine

async def play_engine_move(engine, board, limit, game_id, multipv_req, polyglot_book=None, debug=False, verbose=True):
    if polyglot_book is not None:
        try:
            entry = polyglot_book.weighted_choice(board)
            if verbose: print(f"Book move found: {board.san(entry.move)}")
            return entry.move
        except IndexError:
            pass

    if isinstance(engine, chess.engine.XBoardProtocol):
        result = await engine.play(board, limit, game=game_id)
        if result.move is None:
            raise chess.engine.EngineError("XBoard engine failed to make a move.")
        return result.move

    num_legal_moves = board.legal_moves.count()
    if num_legal_moves == 0:
        raise RuntimeError("Play called on a board with no legal moves.")

    num_pvs_to_display = 0
    actual_multipv_for_engine = None
    if multipv_req > 0:
        num_pvs_to_display = max(1, min(multipv_req, num_legal_moves))
        actual_multipv_for_engine = num_pvs_to_display

    if not verbose:
        result = await engine.play(board, limit, game=game_id, info=chess.engine.INFO_NONE)
        return result.move

    with await engine.analysis(
        board, limit, game=game_id,
        info=chess.engine.INFO_ALL,
        multipv=actual_multipv_for_engine
    ) as analysis:
        infos_to_print = [None] * num_pvs_to_display if num_pvs_to_display > 0 else []
        first_print_cycle = True

        async for info_item in analysis:
            if num_pvs_to_display > 0 and info_item.get("multipv", 0) <= num_pvs_to_display:
                infos_to_print[info_item["multipv"] - 1] = info_item

            can_print_live_analysis = (
                not debug and
                "score" in analysis.info and
                (all(infos_to_print) if num_pvs_to_display > 0 else True)
            )

            if can_print_live_analysis:
                lines_to_clear = (num_pvs_to_display + 1) if num_pvs_to_display > 0 else 1
                if not first_print_cycle:
                    print(f"\u001b[1A\u001b[K" * lines_to_clear, end="")
                else:
                    first_print_cycle = False

                main_score_obj = analysis.info["score"].relative
                main_score_str = (
                    f"Score: {main_score_obj.score()}"
                    if main_score_obj.score() is not None
                    else f"Mate in {main_score_obj.mate()}"
                )
                print(
                    f'{main_score_str}, nodes: {analysis.info.get("nodes", "N/A")}, '
                    f'time: {float(analysis.info.get("time", 0)):.1f}'
                )

        if "pv" not in analysis.info or not analysis.info["pv"]:
            return random.choice(list(board.legal_moves))

        return analysis.info["pv"][0]

async def play_game(
    white_config, black_config,
    multipv_arg, board, time_limit,
    perspective=chess.WHITE, polyglot_book=None,
    debug=False, pgn_file=None, is_chess960=False,
    visualize=True, game_index=1
):
    white_engine = None
    black_engine = None
    game_pgn = chess.pgn.Game()
    
    game_pgn.headers["Event"] = "Human vs Computer" if ("HUMAN" in [white_config, black_config]) else "Engine Match"
    game_pgn.headers["Site"] = "Local"
    if is_chess960: game_pgn.headers["Variant"] = "Chess960"
    if board.fen() != chess.STARTING_FEN:
        game_pgn.headers["FEN"] = board.fen()
        game_pgn.headers["SetUp"] = "1"

    if white_config == "HUMAN":
        game_pgn.headers["White"] = "Human"
    else:
        try:
            if debug and visualize: print(f"Loading White: {white_config['command']}")
            white_engine = await load_engine(
                white_config["command"], white_config["options"], debug=debug
            )
            game_pgn.headers["White"] = white_engine.id.get("name", white_config["command"])
        except Exception as e:
            print(f"Failed to load White engine: {e}")
            return "0-1"

    if black_config == "HUMAN":
        game_pgn.headers["Black"] = "Human"
    else:
        try:
            if debug and visualize: print(f"Loading Black: {black_config['command']}")
            black_engine = await load_engine(
                black_config["command"], black_config["options"], debug=debug
            )
            game_pgn.headers["Black"] = black_engine.id.get("name", black_config["command"])
        except Exception as e:
            print(f"Failed to load Black engine: {e}")
            if white_engine: await white_engine.quit()
            return "1-0"

    node = game_pgn.root()
    game_id = random.random()
    current_result_str = "*"

    try:
        while not board.is_game_over(claim_draw=True):
            
            if visualize:
                print_unicode_board(board, perspective)
            
            is_white = (board.turn == chess.WHITE)
            current_config = white_config if is_white else black_config
            current_engine = white_engine if is_white else black_engine
            player_name = game_pgn.headers["White"] if is_white else game_pgn.headers["Black"]

            if current_config == "HUMAN":
                if visualize: print(f"\n{player_name} to move...")
                move = await get_human_move(board)
            else:
                if visualize: 
                    eng_name = current_engine.id.get('name', 'Engine')
                    print(f"\n{player_name} ({eng_name}) thinking...")
                
                move = await play_engine_move(
                    current_engine, board, time_limit, game_id,
                    multipv_arg, polyglot_book, debug=debug, verbose=visualize
                )

            if visualize:
                print(f"{player_name} plays: {board.san(move)}")
            
            board.push(move)
            node = node.add_main_variation(move)

        current_result_str = board.result(claim_draw=True)
        if board.is_stalemate(): termination = "Stalemate"
        elif board.is_insufficient_material(): termination = "Insufficient material"
        elif board.is_seventyfive_moves(): termination = "75 move rule"
        elif board.is_fivefold_repetition(): termination = "Fivefold repetition"
        elif board.is_checkmate():
            winner = 'White' if board.turn == chess.BLACK else 'Black'
            termination = f"Checkmate, {winner} wins"
        else:
            termination = "Draw"
        
        game_pgn.headers["Result"] = current_result_str
        game_pgn.headers["Termination"] = termination

        if visualize:
            print_unicode_board(board, perspective)
            print(f"Game over. Result: {current_result_str} ({termination})")
        else:
            print(f"Game #{game_index} Finished. Result: {current_result_str}")

    except Exception as e:
        print(f"Error in Game #{game_index}: {e}")
        current_result_str = "*"
    finally:
        if white_engine: await white_engine.quit()
        if black_engine: await black_engine.quit()

    if pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game_pgn.accept(exporter)
        pgn_file.write("\n\n")

    return current_result_str

def parse_engine_options(options_list):
    if not options_list: return {}
    opts = {}
    for opt_str in options_list:
        if '=' not in opt_str: continue
        name, val_str = opt_str.split('=', 1)
        try:
            val = int(val_str)
        except ValueError:
            if val_str.lower() == 'true': val = True
            elif val_str.lower() == 'false': val = False
            else: val = val_str
        opts[name] = val
    return opts


async def safe_game_runner(semaphore, *args, **kwargs):
    async with semaphore:
        return await play_game(*args, **kwargs)

async def main():
    parser = argparse.ArgumentParser(description="Chess Engine Runner")
    
    parser.add_argument("--a", help="Engine A command.")
    parser.add_argument("--engine-a-option", action="append", default=[])
    
    parser.add_argument("--b", help="Engine B command.")
    parser.add_argument("--engine-b-option", action="append", default=[])
    
    parser.add_argument("-human", action="store_true", help="Play as Human vs Engine.")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of games to run in parallel.")
    
    parser.add_argument("--tournament-engines", nargs="+", help="List of engine commands for round-robin.")
    
    parser.add_argument("--pgn", help="Path to PGN file.")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--time", type=float, default=2.0, help="Seconds per move.")
    parser.add_argument("--book", help="Polyglot book path.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--multipv", nargs="?", const=1, default=0, type=int)
    parser.add_argument("--perspective", choices=["white","black"], default="white")
    parser.add_argument("--960", dest="chess960", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    pgn_file = open(args.pgn, "a", encoding="utf-8") if args.pgn else None
    polyglot_book = None
    if args.book:
        try:
            polyglot_book = chess.polyglot.open_reader(args.book)
        except Exception as e:
            print(f"Could not load book: {e}")

    time_limit = chess.engine.Limit(time=args.time)

    opts_a = parse_engine_options(args.engine_a_option)
    opts_b = parse_engine_options(args.engine_b_option)

    # -------------------------------------------------------------------------
    # MODE: Human vs Computer
    # -------------------------------------------------------------------------
    if args.human:
        print("Human vs Computer Mode Enabled.")
        
        white_cfg = "HUMAN"
        black_cfg = "HUMAN"
        perspective = chess.WHITE

        if args.a:
            white_cfg = {"command": args.a, "options": opts_a}
            black_cfg = "HUMAN"
            perspective = chess.BLACK
            print(f"Engine {args.a} is White.")
        elif args.b:
            white_cfg = "HUMAN"
            black_cfg = {"command": args.b, "options": opts_b}
            perspective = chess.WHITE
            print(f"Engine {args.b} is Black.")
        else:
            print("Error: Specify --a (Engine=White) or --b (Engine=Black) for human mode.")
            return

        pos_id = random.randint(0, 959) if args.chess960 else None
        board = chess.Board.from_chess960_pos(pos_id) if args.chess960 else chess.Board(args.fen)
        if args.chess960: print(f"Chess960 ID: {pos_id}")

        await play_game(
            white_cfg, black_cfg, args.multipv, board, time_limit,
            perspective, polyglot_book, args.debug, pgn_file, args.chess960,
            visualize=True, game_index=1
        )
    elif args.tournament_engines:
        configs = []
        for cmd in args.tournament_engines:
            configs.append({"command": cmd, "options": {}, "name": cmd})

        tasks = []
        match_info = []
        semaphore = asyncio.Semaphore(args.concurrency)
        game_counter = 0

        print(f"Starting Tournament: {len(configs)} engines, {args.games} games per pair.")

        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                
                # i vs j
                for _ in range(args.games):
                    game_counter += 1
                    match_info.append((configs[i]['name'], configs[j]['name']))
                    
                    if args.chess960:
                        board = chess.Board.from_chess960_pos(random.randint(0, 959))
                    else:
                        board = chess.Board(args.fen)

                    tasks.append(safe_game_runner(
                        semaphore, configs[i], configs[j], args.multipv, 
                        board, time_limit, chess.WHITE, 
                        polyglot_book, args.debug, pgn_file, args.chess960, 
                        visualize=False, game_index=game_counter
                    ))
                
                # j vs i
                for _ in range(args.games):
                    game_counter += 1
                    match_info.append((configs[j]['name'], configs[i]['name']))

                    if args.chess960:
                        board = chess.Board.from_chess960_pos(random.randint(0, 959))
                    else:
                        board = chess.Board(args.fen)

                    tasks.append(safe_game_runner(
                        semaphore, configs[j], configs[i], args.multipv, 
                        board, time_limit, chess.WHITE, 
                        polyglot_book, args.debug, pgn_file, args.chess960, 
                        visualize=False, game_index=game_counter
                    ))
       
        results = await asyncio.gather(*tasks)
        print("Tournament Finished.")

        stats = {cfg['name']: {'p': 0, 'w': 0, 'd': 0, 'l': 0, 'score': 0.0} for cfg in configs}

        for result, (white, black) in zip(results, match_info):
            stats[white]['p'] += 1
            stats[black]['p'] += 1

            if result == "1-0":
                stats[white]['w'] += 1
                stats[white]['score'] += 1.0
                stats[black]['l'] += 1
            elif result == "0-1":
                stats[black]['w'] += 1
                stats[black]['score'] += 1.0
                stats[white]['l'] += 1
            elif result == "1/2-1/2":
                stats[white]['d'] += 1
                stats[white]['score'] += 0.5
                stats[black]['d'] += 1
                stats[black]['score'] += 0.5

        sorted_stats = sorted(stats.items(), key=lambda item: item[1]['score'], reverse=True)

        print("\n" + "="*65)
        print(f"{'Rank':<5} {'Engine':<25} {'Score':<7} {'Played':<8} {'W':<4} {'D':<4} {'L':<4}")
        print("-" * 65)
        
        for rank, (name, s) in enumerate(sorted_stats, 1):
            print(f"{rank:<5} {name[:24]:<25} {s['score']:<7} {s['p']:<8} {s['w']:<4} {s['d']:<4} {s['l']:<4}")
        
        print("="*65 + "\n")

    elif args.a and args.b:
        cfg_a = {"command": args.a, "options": opts_a}
        cfg_b = {"command": args.b, "options": opts_b}
        
        tasks = []
        semaphore = asyncio.Semaphore(args.concurrency)
        
        print(f"Starting Match: {args.a} vs {args.b}, {args.games} games, Concurrency: {args.concurrency}")
        
        for i in range(args.games):
            if args.chess960:
                board = chess.Board.from_chess960_pos(random.randint(0, 959))
            else:
                board = chess.Board(args.fen)

            tasks.append(safe_game_runner(
                semaphore, cfg_a, cfg_b, args.multipv, 
                board, time_limit, 
                chess.WHITE if args.perspective=="white" else chess.BLACK,
                polyglot_book, args.debug, pgn_file, args.chess960,
                visualize=False, game_index=i+1
            ))

        results = await asyncio.gather(*tasks)
        
        w = results.count("1-0")
        b = results.count("0-1")
        d = results.count("1/2-1/2")
        print(f"\nFinal Results: White Wins: {w}, Black Wins: {b}, Draws: {d}")

    else:
        parser.print_help()

    if pgn_file: pgn_file.close()
    if polyglot_book: polyglot_book.close()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
