{
    'nccl': {
        # lowpp module, the rest is auto-computed based on the layout rule
        'module': 'nccl4py.bindings.nccl',
        'data': {
            'versions': [
                # This is kind of a hack to make Cybind work
                # Call whatever is NOT "0.0.0" to avoid interfering with
                # other versions in Cybind's nccl dir
                ('2.28.0', ),
            ],
            # first one is the master header, the rest is for assisting parsing
            'headers': ['nccl.h'],
            # see the docstring of process_prefix_suffix() for the expected patterns
            'patterns': {
                'function': r'(^nccl)([A-Z].*)',
                'type': r'(^nccl)([A-Z][A-Za-z_].*)_t$|(^nccl)([A-Z].*Id)$',
            },
            # whether parsing the library headers requires also including CUDA headers
            'need_cuda': True,
            # whether checking the driver version is needed before loading the DSO/DLL
            'need_driver': False,
            # whether a Windows module is needed
            'support_win': False,
            # source of docstrings
            # NCCL does not have valid doxygen docstrings...
            'docstrings': None,
            'need_headers_at_build': False,
        },
        # key: actual symbol name
        # alias: define'd or typedef'd name that we prefer users to use
        # TODO: ideally we should turn this relation around, but pycparser
        # only takes the preprocess'd output and does not preserve this info...
        'functions': {
            # Memory helpers
            'ncclMemAlloc': {
                "return": "ptr",
                "except?": 0,
            },
            'ncclMemFree': {
            },

            # Version / IDs
            'ncclGetVersion': {
                "return": "version",
                "except?": -1,
            },
            'ncclGetUniqueId': {
            },

            # Init
            'ncclCommInitRankConfig': {
                # make commId bytes as it is passed by value...
                "pyargs": {"commId": ("BYTES", None),},
                "return": "comm",
                "except?": 0,
            },
            'ncclCommInitRank': {
                # make commId bytes as it is passed by value...
                "pyargs": {"commId": ("BYTES", None),},
                "return": "comm",
                "except?": 0,
            },
            'ncclCommInitAll': {
                "pyargs": {"devlist": "SEQ",},
            },

            # Finalize / destroy / abort
            'ncclCommFinalize': {
            },
            'ncclCommDestroy': {
            },
            'ncclCommAbort': {
            },

            # Split / shrink
            'ncclCommSplit': {
                "return": "newcomm",
                "except?": 0,
            },
            'ncclCommShrink': {
                "pyargs": {"excludeRanksList": "SEQ",},
                "return": "newcomm",
                "except?": 0,
            },

            # Extended init
            'ncclCommInitRankScalable': {
                "pyargs": {"commIds": "NSEQ",},
                "return": "newcomm",
                "except?": 0,
            },

            # Strings / errors
            'ncclGetErrorString': {
                'return': 'STR',
            },
            'ncclGetLastError': {
                'return': 'STR',
            },

            # Queries
            'ncclCommGetAsyncError': {
                "return": "asyncError",
            },
            'ncclCommCount': {
                "return": "count",
                "except?": -1,
            },
            'ncclCommCuDevice': {
                "return": "device",
                "except?": -1,
            },
            'ncclCommUserRank': {
                "return": "rank",
                "except?": -1,
            },

            # Registration
            'ncclCommRegister': {
                "return": "handle",
                "except?": 0,
            },
            'ncclCommDeregister': {
            },

            # Windows
            'ncclCommWindowRegister': {
                "return": "win",
                "except?": 0,
            },
            'ncclCommWindowDeregister': {
            },

            # RedOp
            'ncclRedOpCreatePreMulSum': {
                "return": "op",
            },
            'ncclRedOpDestroy': {
            },

            # Collectives
            'ncclReduce': {
            },
            'ncclBcast': {
            },
            'ncclBroadcast': {
            },
            'ncclAllReduce': {
            },
            'ncclReduceScatter': {
            },
            'ncclAllGather': {
            },
            'ncclAlltoAll': {
            },
            'ncclGather': {
            },
            'ncclScatter': {
            },

            # Point-to-point
            'ncclSend': {
            },
            'ncclRecv': {
            },

            # Group ops
            'ncclGroupStart': {
            },
            'ncclGroupEnd': {
            },
            'ncclGroupSimulateEnd': {
            },
        },
        # use this to control the type apprearance at the lowpp level
        # this is used as WAR for current codegen limitations
        'types': {
            "ncclUniqueId": "AUTO_LOWPP_ARRAY",
            # Note: ncclConfig_t has members appended across versions, while technically
            # it is an ABI-breaking change, as long as we only allocate 1 struct at a time
            # at run time we are fine. NCCL_CONFIG_INITIALIZER needs to be manually
            # implemented in pure Python to initialize this struct.
            "ncclConfig_t": "AUTO_LOWPP_ARRAY",
            "ncclSimInfo_t": "AUTO_LOWPP_ARRAY",
        },
        # map the enum values to their expected dtypes
        # this is very library-specific and needs the library developer to fill in
        # (or by referring to the library manual)
        'attrs': {
        },
        # This is used to patch (processed) enums, in case the generated lowpp enumerator names
        # are not meeting the expectation.
        'enums': {
            "^nccl": "", "^Scalar": "",
        },
    },
}
