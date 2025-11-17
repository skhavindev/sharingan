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
  sharingan-core ui                    # Start web UI
  sharingan-core ui --port 8080        # Start on custom port
  python -m sharingan.cli ui           # Alternative way to run
  sharingan-core --version             # Show version
        """
    )
    
    # Get version from package
    try:
        from sharingan import __version__
        version_str = f'sharingan-core {__version__}'
    except (ImportError, AttributeError):
        version_str = 'sharingan-core 3.0.0'
    
    parser.add_argument(
        '--version',
        action='version',
        version=version_str
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # UI command (alias for run)
    ui_parser = subparsers.add_parser('ui', help='Start Sharingan web UI')
    ui_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    ui_parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    ui_parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    ui_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    # Run command (kept for backwards compatibility)
    run_parser = subparsers.add_parser('run', help='Start Sharingan web UI (alias for ui)')
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
    
    if args.command in ('run', 'ui'):
        from sharingan.ui import run_ui
        run_ui(
            host=args.host,
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser
        )
    else:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)


if __name__ == '__main__':
    main()
