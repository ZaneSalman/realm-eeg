"""Allow ``python -m realm_eeg`` to run the command-line interface."""

from .cli import main

raise SystemExit(main())
