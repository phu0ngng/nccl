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
            'headers': ['nccl_device_expanded.h'],
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
                # commIds is a pointer to a contiguous array of ncclUniqueId.
                "pyargs": {"commIds": ("BYTES", None),},
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
            'ncclCommMemStats': {
                "return": "value",
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
                "return": "win"
            },
            'ncclCommWindowDeregister': {
            },
            # except? is required for C-scalar returns (intptr_t) so Cython
            # can propagate exceptions from check_status(). 0 is fine even
            # though ncclWinGetUserPtr legitimately returns NULL on success
            # (no symmetric support): except? only triggers a PyErr_Occurred()
            # check, it does not treat the sentinel as an error by itself.
            'ncclWinGetUserPtr': {
                "return": "outUserPtr",
                "except?": 0,
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
            'ncclWaitSignal': {
            },
            'ncclSignal': {
            },
            'ncclPutSignal': {
            },

            # Group ops
            'ncclGroupStart': {
            },
            'ncclGroupEnd': {
            },
            'ncclGroupSimulateEnd': {
            },
            'ncclDevCommCreate': {
            },
            'ncclDevCommDestroy': {
            },
            'ncclCommQueryProperties': {
            },
        },
        # use this to control the type apprearance at the lowpp level
        # this is used as WAR for current codegen limitations
        'types': {
            "ncclUniqueId": "AUTO_LOWPP_CLASS",
            "ncclConfig_t": "AUTO_LOWPP_CLASS",
            "ncclSimInfo_t": "AUTO_LOWPP_CLASS",
            "ncclWaitSignalDesc_t": "AUTO_LOWPP_CLASS",
            "ncclCommProperties_t": "AUTO_LOWPP_CLASS",
            "ncclTeam_t": "AUTO_LOWPP_CLASS",
            "ncclMultimemHandle_t": "AUTO_LOWPP_CLASS",
            "ncclTeamRequirements_t": "AUTO_LOWPP_CLASS",
            "ncclDevResourceRequirements_t": "AUTO_LOWPP_CLASS",
            "ncclDevCommRequirements_t": "AUTO_LOWPP_CLASS",
            "ncclGinBarrierHandle_t": "AUTO_LOWPP_CLASS",
            "ncclLsaBarrierHandle_t": "AUTO_LOWPP_CLASS",
            "ncclWindow_vidmem_t": "AUTO_LOWPP_CLASS",
            "ncclDevComm_t": "AUTO_LOWPP_CLASS",
        },
        # map the enum values to their expected dtypes
        # this is very library-specific and needs the library developer to fill in
        # (or by referring to the library manual)
        'attrs': {
        },
        # This is used to patch (processed) enums, in case the generated lowpp enumerator names
        # are not meeting the expectation.
        'enums': {
            "^nccl": "",
            "^Scalar": "",
            "^Stat": "",
            "^GIN_CONNECTION_": "",
        },
    },
}
