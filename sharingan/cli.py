"""Command-line interface for Sharingan."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Sharingan - Semantic Video Understanding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sharingan run              # Start web UI
  sharingan run --port 8080  # Start on custom port
  sharingan --version        # Show version
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Sharingan 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Start Sharingan web UI')
    run_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    run_parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    run_parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    run_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    if args.command == 'run':
        from sharingan.ui import run_ui
        run_ui(
            host=args.host,
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
