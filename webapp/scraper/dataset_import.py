import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scraper.services import (
    _build_program_alias_text,
    _build_structured_page_url,
    _split_program_title_and_placement,
    hash_content,
    infer_general_topic_metadata,
    normalize_whitespace,
    upsert_page_chunks,
    mark_missing_pages_inactive,
)


CANDIDATE_TOPIC_KIND_MAP = {
    'quota': ('admissions_scores', 'Kontenjan ve Puan Tablosu'),
    'tuition': ('tuition', 'Öğrenim Ücretleri'),
    'scholarship': ('scholarships', 'Burs Olanakları'),
    'dorm': ('dormitory', 'Yurt Bilgileri ve Ücretleri'),
    'erasmus': ('international', 'Uluslararası Olanaklar'),
    'bilateral': ('international', 'Uluslararası Olanaklar'),
    'cap_yandal': ('double_major_minor', 'Çift Anadal-Yandal Programları'),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _normalize_source_url(source: dict[str, Any]) -> str:
    return normalize_whitespace(source.get('canonical_url') or source.get('url') or '')


def _get_primary_record(
    chunk: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for record_id in chunk.get('record_ids') or []:
        record = records_by_id.get(record_id)
        if record:
            return record
    return {}


def _enrich_general_chunk_metadata(
    metadata: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(metadata)
    record = record or {}
    payload = record.get('payload') or {}
    record_type = normalize_whitespace(
        metadata.get('record_type') or record.get('record_type') or ''
    )
    entity_name = normalize_whitespace(
        metadata.get('entity_name') or record.get('entity_name') or ''
    )
    unit_name = normalize_whitespace(metadata.get('unit_name') or payload.get('unit_name') or '')
    unit_kind = normalize_whitespace(payload.get('unit_kind', ''))
    faculty = normalize_whitespace(
        metadata.get('faculty')
        or payload.get('faculty_root_name')
        or payload.get('parent_unit_name')
        or payload.get('faculty')
        or ''
    )

    if unit_name:
        metadata['unit_name'] = unit_name
    if faculty:
        metadata['faculty'] = faculty

    program_title = normalize_whitespace(metadata.get('program_title', ''))
    if record_type == 'academic_staff_member':
        if unit_kind == 'faculty':
            if not metadata.get('faculty') and unit_name:
                metadata['faculty'] = unit_name
        else:
            program_title = program_title or unit_name
            if not metadata.get('faculty') and unit_kind == 'department' and faculty:
                metadata['faculty'] = faculty
        staff_role = normalize_whitespace(payload.get('staff_role', ''))
        staff_title = normalize_whitespace(payload.get('staff_title', ''))
        if staff_role:
            metadata['staff_role'] = staff_role
        if staff_title:
            metadata['staff_title'] = staff_title
    elif record_type.startswith('department_'):
        program_title = program_title or entity_name or unit_name

    if not program_title and metadata.get('source_group') == 'department':
        program_title = entity_name or unit_name

    if program_title:
        metadata['program_title'] = program_title
        metadata.setdefault(
            'program_alias_text',
            _build_program_alias_text(program_title=program_title),
        )

    return metadata


def _derive_general_page_metadata(
    source: dict[str, Any],
    source_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'source_id': source['source_id'],
        'source_group': source.get('source_group', ''),
        'source_variant': source.get('source_variant', ''),
        'candidate_page_kind': source.get('candidate_page_kind'),
        'section_title': source.get('title') or '',
        'source_url': _normalize_source_url(source),
        'host': source.get('host', ''),
        'import_origin': 'clean_dataset',
    }
    if source.get('program_level'):
        metadata['program_level'] = source['program_level']

    record_types = {
        chunk.get('metadata', {}).get('record_type')
        for chunk in source_chunks
        if chunk.get('metadata', {}).get('record_type')
    }
    if 'academic_staff_member' in record_types:
        metadata['kind'] = 'main_site_staff_page'
        metadata['staff_count'] = sum(
            1
            for chunk in source_chunks
            if chunk.get('metadata', {}).get('record_type') == 'academic_staff_member'
        )
    else:
        topic_kind = CANDIDATE_TOPIC_KIND_MAP.get(source.get('candidate_page_kind', ''))
        if topic_kind:
            metadata['kind'] = 'candidate_topic_page'
            metadata['topic'], metadata['topic_label'] = topic_kind
        else:
            metadata['kind'] = 'main_site_page'
            topic_metadata = infer_general_topic_metadata(
                _normalize_source_url(source),
                source.get('title') or '',
            )
            for key, value in topic_metadata.items():
                metadata.setdefault(key, value)
    return metadata


def _apply_general_page_context(
    page_metadata: dict[str, Any],
    chunk_metadatas: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    page_metadata = dict(page_metadata)
    for field in ('program_title', 'program_alias_text', 'faculty', 'unit_name'):
        if page_metadata.get(field):
            continue
        for chunk_metadata in chunk_metadatas:
            value = normalize_whitespace(chunk_metadata.get(field, ''))
            if value:
                page_metadata[field] = value
                break

    if (
        not page_metadata.get('program_title')
        and source.get('source_group') == 'department'
        and source.get('title')
    ):
        program_title = normalize_whitespace(source['title'].split(' - ', 1)[0])
        if program_title:
            page_metadata['program_title'] = program_title
            page_metadata.setdefault(
                'program_alias_text',
                _build_program_alias_text(program_title=program_title),
            )

    return page_metadata


def _build_general_chunk_metadata(
    page_metadata: dict[str, Any],
    chunk: dict[str, Any],
    index: int,
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = _enrich_general_chunk_metadata(
        {
        **page_metadata,
        **(chunk.get('metadata') or {}),
        'chunk_id': chunk.get('chunk_id'),
        'chunk_type': chunk.get('chunk_type'),
        'language': chunk.get('language', 'tr'),
        'record_ids': chunk.get('record_ids', []),
        'import_chunk_index': index,
        },
        _get_primary_record(chunk, records_by_id),
    )
    metadata.setdefault('kind', page_metadata.get('kind', 'main_site_page'))
    return metadata


def _build_general_pages(dataset_root: Path) -> list[dict[str, Any]]:
    sources = {
        source['source_id']: source
        for source in _read_jsonl(dataset_root / 'acibadem_output' / 'sources_clean.jsonl')
    }
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in _read_jsonl(dataset_root / 'acibadem_output' / 'chunks_clean.jsonl'):
        chunks_by_source[chunk['source_id']].append(chunk)
    records_by_id = {
        record['record_id']: record
        for record in _read_jsonl(dataset_root / 'acibadem_output' / 'records_clean.jsonl')
    }

    pages: list[dict[str, Any]] = []
    for source_id, source in sources.items():
        source_chunks = chunks_by_source.get(source_id, [])
        if not source_chunks:
            continue
        base_page_metadata = _derive_general_page_metadata(source, source_chunks)
        chunk_metadatas = [
            _build_general_chunk_metadata(base_page_metadata, chunk, index, records_by_id)
            for index, chunk in enumerate(source_chunks)
            if normalize_whitespace(chunk.get('text', ''))
        ]
        page_metadata = _apply_general_page_context(base_page_metadata, chunk_metadatas, source)
        content_text = normalize_whitespace(source.get('text', '')) or '\n\n'.join(
            normalize_whitespace(chunk.get('text', '')) for chunk in source_chunks
        )
        chunk_rows = [
            {
                'text': normalize_whitespace(chunk.get('text', '')),
                'metadata': _build_general_chunk_metadata(page_metadata, chunk, index, records_by_id),
                'chunk_index': index,
            }
            for index, chunk in enumerate(source_chunks)
            if normalize_whitespace(chunk.get('text', ''))
        ]
        if not chunk_rows:
            continue
        pages.append(
            {
                'source': 'main_site',
                'url': _normalize_source_url(source),
                'title': source.get('title') or _normalize_source_url(source),
                'text': content_text,
                'raw_html': source.get('text', ''),
                'metadata': page_metadata,
                'chunks': chunk_rows,
            }
        )
    return pages


def _build_quota_page(record: dict[str, Any], source_url: str) -> dict[str, Any]:
    payload = record.get('payload') or {}
    raw_label = payload.get('Fakülte/Bölüm Adı') or record.get('entity_name') or ''
    program_title, placement_label, placement_type = _split_program_title_and_placement(raw_label)
    score_type = normalize_whitespace(payload.get('Puan Türü', ''))
    quota = normalize_whitespace(payload.get('Kontenjan', ''))
    top_score = normalize_whitespace(payload.get('Tavan Puan', ''))
    top_rank = normalize_whitespace(payload.get('Başarı Sırası', '') or payload.get('Tavan Başarı Sırası', ''))
    base_score = normalize_whitespace(payload.get('Taban Puan', ''))
    base_rank = normalize_whitespace(payload.get('Taban Başarı Sırası', ''))
    faculty = normalize_whitespace(
        payload.get('faculty')
        or payload.get('Fakülte')
        or payload.get('Akademik Birim')
        or ''
    )
    metadata = {
        'kind': 'structured_admissions_score',
        'topic': 'admissions_scores',
        'topic_label': 'Kontenjan ve Puan Tablosu',
        'record_type': 'quota_row',
        'source_group': 'quota',
        'section_title': 'Kontenjan ve Puan',
        'program_title': program_title or placement_label,
        'program_alias_text': _build_program_alias_text(
            program_title=program_title or placement_label,
            placement_label=placement_label,
            placement_type=placement_type,
        ),
        'placement_label': placement_label,
        'placement_type': placement_type,
        'faculty': faculty,
        'admission_level': record.get('program_level') or '',
        'score_type': score_type,
        'quota': quota,
        'top_score': top_score,
        'top_rank': top_rank,
        'base_score': base_score,
        'base_rank': base_rank,
        'source_url': source_url,
        'source_id': record.get('source_id'),
        'year': record.get('year'),
        'entity_name': record.get('entity_name', ''),
        'import_origin': 'clean_dataset',
    }
    lines = [
        f'Program: {metadata["program_title"]}',
        f'Yerleşim: {placement_label or "-"}',
        f'Yerleşim Türü: {placement_type or "-"}',
        f'Akademik Birim: {faculty or "-"}',
        f'Puan Türü: {score_type or "-"}',
        f'Kontenjan: {quota or "-"}',
        f'Tavan Puan: {top_score or "-"}',
        f'Tavan Başarı Sırası: {top_rank or "-"}',
        f'Taban Puan: {base_score or "-"}',
        f'Taban Başarı Sırası: {base_rank or "-"}',
    ]
    text = '\n'.join(lines)
    return {
        'source': 'structured',
        'url': _build_structured_page_url(
            'admissions-score',
            record.get('program_level') or '',
            faculty,
            placement_label,
            score_type,
        ),
        'title': f'{placement_label} - Kontenjan ve Puan',
        'text': text,
        'raw_html': json.dumps(record, ensure_ascii=False),
        'metadata': metadata,
        'chunks': [
            {
                'text': text,
                'metadata': metadata,
                'chunk_index': 0,
            }
        ],
    }


def _build_fee_page(record: dict[str, Any], source_url: str) -> dict[str, Any]:
    payload = record.get('payload') or {}
    pricing = payload.get('pricing') or {}
    program_title = normalize_whitespace(payload.get('program') or record.get('entity_name') or '')
    faculty = normalize_whitespace(payload.get('faculty', ''))
    fee_full = normalize_whitespace(pricing.get('ucretli', ''))
    fee_25 = normalize_whitespace(pricing.get('indirim_25', ''))
    fee_50 = normalize_whitespace(pricing.get('indirim_50', ''))
    fee_kav_support = normalize_whitespace(pricing.get('kav_destek', '') or pricing.get('kav_support', ''))
    notes = ' '.join(normalize_whitespace(note) for note in payload.get('notes', []) if note)
    metadata = {
        'kind': 'structured_admissions_fee',
        'topic': 'tuition',
        'topic_label': 'Öğrenim Ücretleri',
        'record_type': 'tuition_fee',
        'source_group': 'tuition',
        'section_title': 'Öğrenim Ücreti',
        'program_title': program_title,
        'program_alias_text': _build_program_alias_text(program_title=program_title),
        'faculty': faculty,
        'admission_level': payload.get('level') or record.get('program_level') or '',
        'fee_year': record.get('year') or '',
        'fee_full': fee_full,
        'fee_25': fee_25,
        'fee_50': fee_50,
        'fee_kav_support': fee_kav_support,
        'notes': notes,
        'source_url': source_url,
        'source_id': record.get('source_id'),
        'entity_name': record.get('entity_name', ''),
        'import_origin': 'clean_dataset',
    }
    lines = [
        f'Program: {program_title}',
        f'Akademik Birim: {faculty or "-"}',
        f'Ücretli: {fee_full or "-"}',
        f'%25 İndirimli Ücret: {fee_25 or "-"}',
        f'%50 İndirimli Ücret: {fee_50 or "-"}',
        f'İlave %25 KAV Destek Burslu Ücret: {fee_kav_support or "-"}',
    ]
    if notes:
        lines.append(f'Notlar: {notes}')
    text = '\n'.join(lines)
    return {
        'source': 'structured',
        'url': _build_structured_page_url(
            'admissions-fee',
            payload.get('level') or record.get('program_level') or '',
            faculty,
            program_title,
        ),
        'title': f'{program_title} - Öğrenim Ücreti',
        'text': text,
        'raw_html': json.dumps(record, ensure_ascii=False),
        'metadata': metadata,
        'chunks': [
            {
                'text': text,
                'metadata': metadata,
                'chunk_index': 0,
            }
        ],
    }


def _build_structured_pages(dataset_root: Path) -> list[dict[str, Any]]:
    sources = {
        source['source_id']: source
        for source in _read_jsonl(dataset_root / 'acibadem_output' / 'sources_clean.jsonl')
    }
    pages: list[dict[str, Any]] = []
    for record in _read_jsonl(dataset_root / 'acibadem_output' / 'records_clean.jsonl'):
        source_url = _normalize_source_url(sources.get(record.get('source_id'), {}))
        if record.get('record_type') == 'quota_row':
            pages.append(_build_quota_page(record, source_url))
        elif record.get('record_type') == 'tuition_fee':
            pages.append(_build_fee_page(record, source_url))
    return pages


def _build_program_overview_text(
    source: dict[str, Any],
    summary: dict[str, Any] | None,
) -> str:
    lines = [
        f'Program: {source.get("program", "")}',
        f'Fakülte: {source.get("faculty", "")}',
        f'Müfredat Yılı: {source.get("curriculum_year", "")}',
    ]
    if summary:
        if summary.get('period_count'):
            lines.append(f'Dönem Sayısı: {summary["period_count"]}')
        if summary.get('course_count'):
            lines.append(f'Toplam Ders Sayısı: {summary["course_count"]}')
        if summary.get('total_ects_sum'):
            lines.append(f'Toplam AKTS: {summary["total_ects_sum"]}')
        for period_label, total in (summary.get('reported_period_totals') or {}).items():
            lines.append(f'- {period_label}: {total} AKTS')
    return '\n'.join(line for line in lines if normalize_whitespace(line))


def _build_semester_plan_text(
    source: dict[str, Any],
    period_label: str,
    rows: list[dict[str, Any]],
) -> str:
    child_rows_by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get('course_row_kind') == 'group_child':
            child_rows_by_parent[
                (
                    normalize_whitespace(row.get('group_parent_code', '')),
                    normalize_whitespace(row.get('group_parent_name', '')),
                )
            ].append(row)

    lines = [
        f'Program: {source.get("program", "")}',
        f'Fakülte: {source.get("faculty", "")}',
        f'Müfredat Yılı: {source.get("curriculum_year", "")}',
        f'Dönem: {period_label}',
    ]
    for row in sorted(rows, key=lambda item: (item.get('course_order') or 9999, item.get('course_name', ''))):
        kind = row.get('course_row_kind')
        if kind == 'group_child':
            continue
        line = (
            f'- {normalize_whitespace(row.get("course_code", ""))} '
            f'{normalize_whitespace(row.get("course_name", ""))} | '
            f'AKTS: {normalize_whitespace(str(row.get("ects", "")))} | '
            f'Saat: {normalize_whitespace(row.get("hours_raw", "")) or "-"} | '
            f'{"Zorunlu" if row.get("is_mandatory") else "Seçmeli"}'
        ).strip()
        if row.get('teaching_method'):
            line += f' | Öğretim: {normalize_whitespace(row["teaching_method"])}'
        children = child_rows_by_parent.get(
            (
                normalize_whitespace(row.get('course_code', '')),
                normalize_whitespace(row.get('course_name', '')),
            ),
            [],
        )
        if children:
            child_labels = [
                f'{normalize_whitespace(child.get("course_code", ""))} {normalize_whitespace(child.get("course_name", ""))}'.strip()
                for child in sorted(children, key=lambda item: (item.get('course_order') or 9999, item.get('course_name', '')))
            ]
            line += f' | Seçenekler: {"; ".join(label for label in child_labels if label)}'
        lines.append(line)
    return '\n'.join(line for line in lines if normalize_whitespace(line))


def _build_bologna_pages(dataset_root: Path) -> list[dict[str, Any]]:
    source_rows = _read_jsonl(dataset_root / 'bologna_courses' / 'sources.jsonl')
    record_rows = _read_jsonl(dataset_root / 'bologna_courses' / 'records.jsonl')
    summary_rows = _read_json(dataset_root / 'bologna_courses' / 'summary.json').get('programs', [])
    records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in record_rows:
        records_by_source[record['source_id']].append(record)
    summary_by_program = {
        (
            normalize_whitespace(row.get('faculty', '')),
            normalize_whitespace(row.get('program', '')),
            normalize_whitespace(row.get('curriculum_year', '')),
        ): row
        for row in summary_rows
    }

    pages: list[dict[str, Any]] = []
    for source in source_rows:
        source_id = source['source_id']
        source_url = _normalize_source_url(source)
        records = records_by_source.get(source_id, [])
        if not records:
            continue
        page_metadata = {
            'kind': 'bologna_program_page',
            'source_group': 'bologna_curriculum',
            'faculty': source.get('faculty', ''),
            'program_title': source.get('program', ''),
            'curriculum_year': source.get('curriculum_year', ''),
            'period_type': source.get('period_type', ''),
            'source_url': source_url,
            'source_id': source_id,
            'import_origin': 'clean_dataset',
        }
        summary = summary_by_program.get(
            (
                normalize_whitespace(source.get('faculty', '')),
                normalize_whitespace(source.get('program', '')),
                normalize_whitespace(source.get('curriculum_year', '')),
            )
        )
        overview_metadata = {
            **page_metadata,
            'record_type': 'bologna_program_overview',
            'chunk_level': 'program_overview',
            'section_title': 'Program Özeti',
            'total_ects_sum': (summary or {}).get('total_ects_sum'),
            'period_count': (summary or {}).get('period_count'),
            'course_count': (summary or {}).get('course_count'),
        }
        overview_text = _build_program_overview_text(source, summary)
        pages.append(
            {
                'source': 'bologna',
                'url': f'{source_url}#overview',
                'title': f'{source.get("program", "")} - Program Özeti',
                'text': overview_text,
                'raw_html': json.dumps(summary or source, ensure_ascii=False),
                'metadata': overview_metadata,
                'chunks': [
                    {
                        'text': overview_text,
                        'metadata': overview_metadata,
                        'chunk_index': 0,
                    }
                ],
            }
        )

        rows_by_period: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            rows_by_period[int(record.get('period_number') or 0)].append(record)
        for period_number, period_rows in sorted(rows_by_period.items()):
            if not period_number:
                continue
            period_label = normalize_whitespace(period_rows[0].get('period_label', '')) or f'{period_number}. dönem'
            semester_metadata = {
                **page_metadata,
                'record_type': 'bologna_semester_plan',
                'chunk_level': 'semester_plan',
                'section_title': period_label,
                'period_number': period_number,
                'period_label': period_label,
            }
            semester_text = _build_semester_plan_text(source, period_label, period_rows)
            pages.append(
                {
                    'source': 'bologna',
                    'url': f'{source_url}#period-{period_number}',
                    'title': f'{source.get("program", "")} - {period_label}',
                    'text': semester_text,
                    'raw_html': json.dumps(
                        {
                            'source_id': source_id,
                            'period_number': period_number,
                            'period_label': period_label,
                        },
                        ensure_ascii=False,
                    ),
                    'metadata': semester_metadata,
                    'chunks': [
                        {
                            'text': semester_text,
                            'metadata': semester_metadata,
                            'chunk_index': 0,
                        }
                    ],
                }
            )
    return pages


def _upsert_pages(pages: list[dict[str, Any]], *, force_refresh: bool) -> tuple[int, int]:
    page_count = 0
    chunk_count = 0
    for page in pages:
        upsert_page_chunks(
            source=page['source'],
            url=page['url'],
            title=page['title'],
            text=page['text'],
            raw_html=page['raw_html'],
            metadata=page['metadata'],
            chunks=page['chunks'],
            force_refresh=force_refresh,
        )
        page_count += 1
        chunk_count += len(page['chunks'])
    return page_count, chunk_count


def import_acibadem_dataset(dataset_root: str, *, force_refresh: bool = False) -> dict[str, int]:
    root = Path(dataset_root)
    main_site_pages = _build_general_pages(root)
    structured_pages = _build_structured_pages(root)
    bologna_pages = _build_bologna_pages(root)

    general_page_count, general_chunk_count = _upsert_pages(
        main_site_pages,
        force_refresh=force_refresh,
    )
    structured_page_count, structured_chunk_count = _upsert_pages(
        structured_pages,
        force_refresh=force_refresh,
    )
    bologna_page_count, bologna_chunk_count = _upsert_pages(
        bologna_pages,
        force_refresh=force_refresh,
    )

    mark_missing_pages_inactive('main_site', [page['url'] for page in main_site_pages])
    mark_missing_pages_inactive('structured', [page['url'] for page in structured_pages])
    mark_missing_pages_inactive('bologna', [page['url'] for page in bologna_pages])

    return {
        'pages': general_page_count + structured_page_count + bologna_page_count,
        'chunks': general_chunk_count + structured_chunk_count + bologna_chunk_count,
        'main_site_pages': general_page_count,
        'structured_pages': structured_page_count,
        'bologna_pages': bologna_page_count,
        'force_refresh': int(force_refresh),
        'manifest_hash': hash_content(
            json.dumps(
                {
                    'general_pages': len(main_site_pages),
                    'structured_pages': len(structured_pages),
                    'bologna_pages': len(bologna_pages),
                },
                sort_keys=True,
            )
        ),
    }
